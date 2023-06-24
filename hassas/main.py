"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from feddropserver import FedDropServer
from fedavgserver import FedAvgServer
from fedproxserver import FedProxServer
from afdserver import AFDServer
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data
from optimizer.pgd import PerturbedGradientDescent
import copy
import json

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def save_model(server, args, round):
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.metrics_dir)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    filename = ckpt_path + 'round.txt'
    with open(filename, "w") as f:
        f.write(str(round))

    save_path1, save_path2 = server.save_model(os.path.join(ckpt_path, '{}'.format(args.model)))
    print('High Model saved in path: %s' % save_path1)
    print('Low Model saved in path: %s' % save_path2)

def main():

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    os.makedirs(os.path.dirname(args.metrics_dir), exist_ok=True)
    if args.optimizer == 'feddrop':
        if args.dataset == 'synthetic':
            layer_names = ['dense1/kernel:0']
        elif 'femnist' in args.dataset or args.dataset == 'mnist' or args.dataset == 'fmnist' or args.dataset == 'femnist_skewed' or args.dataset == 'cifar':
            layer_names  = ['conv1/kernel:0', 'conv_last/kernel:0', 'dense1/kernel:0']
        elif 'celeba' in args.dataset:
            layer_names  = ['conv_batch1/kernel:0','conv_batch2/kernel:0','conv_batch3/kernel:0', 'conv_last_batch/kernel:0']
        elif args.dataset == 'shakespeare':
            layer_names = ["rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0", "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0"]
        elif args.dataset == "sent140":
            layer_names = ["rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0", "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0" , "dense1/kernel:0"]
        else:
            pass

        p = args.model_droprate
        tf.reset_default_graph()
        client_model_high = ClientModel(args.seed, *model_params, 'H')
        masks = create_masks(client_model_high, layer_names, args, p)
        client_model_low = ClientModel(args.seed, *model_params, 'L', masks)

        # Create server
        server = FedDropServer(client_model_high, client_model_low, masks)
        # Create clients
        clients = setup_clients(args.dataset, client_model_high, client_model_low, args.use_val_set, args.droprate)
        client_ids, client_groups, client_num_samples, client_types = server.get_clients_info(clients)
        print('Clients in Total: %d' % len(clients))
        print('Number of Slow Clients: %d' % (int(args.droprate * clients_per_round)))
        print('Number of Fast Clients: %d' % (clients_per_round - int(args.droprate * clients_per_round)))

        # Initial status
        print('--- Random Initialization ---')
        stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, client_types, args)
        sys_writer_fn = get_sys_writer_function(args)
        print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
        
        # Simulate training
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))
            
            # Select clients to train this round
            server.select_clients(i, online(clients), num_clients=clients_per_round, droprate=args.droprate)
            c_ids, c_groups, c_num_samples, c_types = server.get_clients_info(server.selected_clients)

            # Simulate server model training on selected clients' data
            sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch, lr=args.lr, actorgrad=args.actorgrad)
            sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples, c_types)
            
            # Update server model
            if args.tech == 'clt':
                server.update_model_clt((i + 1))
            else:
                server.update_model()

            # Test model
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

            if ((not args.persist and (i + 1) % args.pround == 0) or (args.persist and (i + 1) == args.pround)) and args.actorgrad == 'act':
                print("Updating CNN masks")
                server.updated_cnn_mask()
                # storing 1 value for each client in a list
                print("Pruning based on Act")
                averaged_acts = get_averaged_acts(clients, server)

                layer_names = ['dense1/kernel:0']
                client_model_high = server.client_model_high
                client_model_high_params = copy.copy(server.model_high)

                dense_mask = create_intelligent_masks_activation(client_model_high, averaged_acts, layer_names, args, p)
                server.masks['dense1/kernel:0'] = dense_mask['dense1/kernel:0']
                client_model_low_params = server.convert_weights_to_smaller(client_model_high_params, server.masks)
                server.client_model_low = ClientModel(args.seed, *model_params, 'L', server.masks)
                server.client_model_low.set_params(client_model_low_params)

                server.model_low = client_model_low_params
                for c in clients:
                    if c.type == 'L':
                        c._model = server.client_model_low
            if (i + 1) % 50 == 0 and args.save_model:
                # Save server model
                save_model(server, args, i+1)

        # Close models
        server.close_model()

    elif args.optimizer == 'afd':
        if args.dataset == 'synthetic':
            layer_names = ['dense1/kernel:0']
        elif 'femnist' in args.dataset or args.dataset == 'mnist' or args.dataset == 'fmnist' or args.dataset == 'femnist_skewed':
            layer_names  = ['conv1/kernel:0', 'conv_last/kernel:0', 'dense1/kernel:0']
        elif 'celeba' in args.dataset:
            layer_names  = ['conv_batch1/kernel:0','conv_batch2/kernel:0','conv_batch3/kernel:0', 'conv_last_batch/kernel:0']
        elif args.dataset == 'shakespeare':
            layer_names = ["rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0", "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0"]
        elif args.dataset == "sent140":
            layer_names = ["dense1/kernel:0"]
        else:
            pass

        p = args.model_droprate
        tf.reset_default_graph()
        client_model_high = ClientModel(args.seed, *model_params, 'H')
        masks = create_masks(client_model_high, layer_names, args, p)
        client_model_low = ClientModel(args.seed, *model_params, 'L', masks)
        scoremap = {k:np.ones(masks[k][1], dtype = float) for k in masks}

        # Create server
        server = AFDServer(client_model_high, client_model_low, masks)
        # Create clients
        clients = setup_clients(args.dataset, client_model_low, client_model_low, args.use_val_set, args.droprate)
        client_ids, client_groups, client_num_samples, client_types = server.get_clients_info(clients)
        print('Clients in Total: %d' % len(clients))
        print('Number of Slow Clients: %d' % (int(args.droprate * clients_per_round)))
        print('Number of Fast Clients: %d' % (clients_per_round - int(args.droprate * clients_per_round)))

        # Initial status
        print('--- Random Initialization ---')
        stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, client_types, args)
        sys_writer_fn = get_sys_writer_function(args)
        print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
        prev_loss = 0
        update_subset_model = False
        # Simulate training
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))                
            
            # Select clients to train this round
            server.select_clients(i, online(clients), num_clients=clients_per_round, droprate=args.droprate)
            c_ids, c_groups, c_num_samples, c_types = server.get_clients_info(server.selected_clients)

            # Simulate server model training on selected clients' data
            sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch, lr=args.lr, actorgrad=args.actorgrad)
            sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples, c_types)
            
            current_loss = get_avg_train_loss(server, server.selected_clients, c_num_samples)
            if (i + 1) > 1:
                if current_loss < prev_loss:
                    score = (prev_loss - current_loss) / prev_loss
                    for k in server.masks:
                        indices = server.masks[k][2]
                        scoremap[k][indices] += score
                    update_subset_model = False
                    print('Same Model')
                else:
                    update_subset_model = True
            prev_loss = current_loss

            # Update server model
            if args.tech == 'clt':
                server.update_model_clt((i + 1))
            else:
                server.update_model()

            if update_subset_model:
                print('Weighted Random Selection')
                for k in server.masks:
                    N = int(server.masks[k][1])
                    dropN = int(server.masks[k][1]*p)
                    new_mask_indices = weighted_sample_without_replacement(server.masks[k][2], scoremap[k], N-dropN)
                    new_mask = np.zeros(N, dtype=bool)
                    new_mask[new_mask_indices] = 1
                    server.masks[k][2] = new_mask

                client_model_high = server.client_model_high
                client_model_high_params = copy.copy(server.model_high)
                client_model_low_params = server.convert_weights_to_smaller(client_model_high_params, server.masks)
                # server.client_model_low = ClientModel(args.seed, *model_params, 'L', server.masks)
                server.client_model_low.set_params(client_model_low_params)

                server.model_low = client_model_low_params
                for c in clients:
                    if c.type == 'L':
                        c._model = server.client_model_low



            # Test model
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

        server.close_model()

    elif args.optimizer == 'fedavg':
        # Create client model, and share params with server model
        tf.reset_default_graph()
        client_model = ClientModel(args.seed, *model_params)

        # Create server
        server = FedAvgServer(client_model)
        # Create clients
        clients = setup_clients(args.dataset, client_model, client_model, args.use_val_set, args.droprate)
        client_ids, client_groups, client_num_samples, client_types = server.get_clients_info(clients)
        print('Clients in Total: %d' % len(clients))
        print('Training Clients: %d' % (clients_per_round - int(args.droprate * clients_per_round)))

        # Initial status
        print('--- Random Initialization ---')
        stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, client_types, args)
        sys_writer_fn = get_sys_writer_function(args)
        print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

        # Simulate training
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            # Select clients to train this round
            server.select_clients(i, online(clients), num_clients=clients_per_round, droprate=args.droprate)
            c_ids, c_groups, c_num_samples, c_types = server.get_clients_info(server.selected_clients)
            # Simulate server model training on selected clients' data
            sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
            sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples, c_types)
            
            # Update server model
            server.update_model()

            # Test model
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
        
        if args.save_model:
            # Save server model
            ckpt_path = os.path.join('checkpoints', args.dataset)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
            print('Model saved in path: %s' % save_path)

        # Close models
        server.close_model()

    elif args.optimizer == 'fedprox':
        ################################################################################
        inner_opt = None
        inner_opt = PerturbedGradientDescent(model_params[0], 1)
        # Create client model, and share params with server model
        tf.reset_default_graph()
        client_model = ClientModel(args.seed, *model_params, 'H', None, inner_opt)
        ################################################################################

        # Create server
        server = FedProxServer(client_model)

        # Create clients
        clients = setup_clients(args.dataset, client_model, client_model, args.use_val_set, args.droprate)
        client_ids, client_groups, client_num_samples, client_types = server.get_clients_info(clients)
        print('Clients in Total: %d' % len(clients))
        print('Number of Slow Clients: %d' % (int(args.droprate * clients_per_round)))
        print('Number of Fast Clients: %d' % (clients_per_round - int(args.droprate * clients_per_round)))

        # Initial status
        print('--- Random Initialization ---')
        stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, client_types, args)
        sys_writer_fn = get_sys_writer_function(args)
        print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

        # Simulate training
        for i in range(num_rounds):

            ################################################################################
            inner_opt.set_params(server.model, server.client_model)
            ################################################################################

            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            # Select clients to train this round
            server.select_clients(i, online(clients), num_clients=clients_per_round, droprate=args.droprate)
            c_ids, c_groups, c_num_samples, c_types = server.get_clients_info(server.selected_clients)


            # Simulate server model training on selected clients' data
            sys_metrics = server.train_model(i=i, num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch, clients_per_round=clients_per_round, droprate=args.droprate)
            sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples, c_types)
            
            # Update server model
            server.update_model()

            
            # Test model
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

        if args.save_model:
            # Save server model
            ckpt_path = os.path.join('checkpoints', args.dataset)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
            print('Model saved in path: %s' % save_path)

        # Close models
        server.close_model()


def get_avg_train_loss(server, clients, num_samples):
    metrics = server.test_client_model(clients, set_to_use='train')
    ordered_metric = [metrics[c]['loss'] for c in sorted(metrics)]
    ordered_weights = [num_samples[c] for c in sorted(num_samples)]
    avg_loss = np.average(ordered_metric, weights=ordered_weights)
    return round(avg_loss, 4)

def weighted_sample_without_replacement(population, weights, k=1):
    weights = list(weights)
    positions = range(len(population))
    indices = []
    while True:
        needed = k - len(indices)
        if not needed:
            break
        for i in random.choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0.0
                indices.append(i)
    return indices

def get_averaged_acts(clients, server):
    clients_activations = []
    hactivations = []
    for c in clients:
        if len(c.activations) > 0:
            if c.type == 'H':
                clients_activations.append(c.activations[0])
                hactivations.append(c.activations[0])
            else:
                act_mask = server.masks['dense1/kernel:0'][2]
                clients_activations.append(server.update_to_org_activations(c.activations[0], act_mask))
        c.activations = []

    indices = server.masks["dense1/kernel:0"][2]
    haveraged_acts = np.mean(np.array(hactivations), axis=0)
    allaveraged_acts = np.mean(np.array(clients_activations), axis=0)
    averaged_acts = np.array([ allaveraged_acts[i] if indices[i] else haveraged_acts[i] for i in range(len(indices)) ])
    return averaged_acts

def online(clients):
    """We assume all users are always online."""
    return clients

def create_clients(users, groups, train_data, test_data, model_high, model_low, droprate):
    random.shuffle(users)
    if len(groups) == 0:
        groups = [[] for _ in users]

    userhigh = users[int(len(users)*droprate):]
    groupshigh = groups[int(len(users)*droprate):]
    userlow = users[0:int(len(users)*droprate)]
    groupslow = groups[0:int(len(users)*droprate)]

    clients = [Client(u, g, train_data[u], test_data[u], model_high, 'H') for u, g in zip(userhigh, groupshigh)]
    clients = clients + [Client(u, g, train_data[u], test_data[u], model_low, 'L') for u, g in zip(userlow, groupslow)]
    return clients

def setup_clients(dataset, model_high=None, model_low=None, use_val_set=False, droprate=0):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    clients = create_clients(users, groups, train_data, test_data, model_high, model_low, droprate)
    return clients

def get_stat_writer_function(ids, groups, num_samples, types, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, types, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))
    return writer_fn

def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples, types):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, types, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn

def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, args, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, args, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)

def print_metrics(metrics, weights, args, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    
    os.makedirs(os.path.dirname(args.metrics_dir), exist_ok=True)
    filename = args.metrics_dir + 'print_data.txt'
    append_write = 'a' if os.path.exists(filename) else 'w'
    with open(filename, append_write) as f:
        for metric in metric_names:
            ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
            print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
                % (prefix + metric,
                    np.average(ordered_metric, weights=ordered_weights),
                    np.percentile(ordered_metric, 10),
                    np.percentile(ordered_metric, 50),
                    np.percentile(ordered_metric, 90)), file=f)

def get_dropout_layer_info(model, layer_names):
    i = 0
    info = {}
    with model.graph.as_default():
        tvars = tf.trainable_variables()
        for layer in tvars:
            if layer.name in layer_names:
                temp = []
                temp.append(i)
                if 'conv' in layer.name:
                    temp.append(int(layer.shape[3]))
                elif 'rnn' in layer.name:
                    temp.append((int(layer.shape[0]), int(layer.shape[1])))
                else:
                    temp.append(int(layer.shape[1]))
                info[layer.name] = temp
            i += 1
    return info

def create_masks(model, layer_names, args, p):
    info = get_dropout_layer_info(model, layer_names)
    for i in info:
        if "rnn" in i:
            N = info[i][1][1]
            m = np.ones(N, dtype=bool)
            tot = int(N/4)
            dropN = int(N/4 * p)
            dropInd = np.random.choice(tot, dropN, replace=False)
            masks = []
            if "cell_0" in i:
                emb = info[i][1][0]
                emb_m = np.ones(emb, dtype=bool)
                embdropInd = np.random.choice(emb, dropN, replace=False)
                emb_m[embdropInd] = 0
                masks.append(emb_m)

            if "cell_1" in i: #last rnn layer rnn->dense
                N = tot
                m2 = np.ones(N, dtype=bool)
            else:
                N = tot * 2
                m2 = np.ones(N, dtype=bool)

            for di in dropInd:
                d  = di * 4
                np.put(m, [d, d+1, d+2, d+3], 0)
                masks.append(m)
                if "cell_1" in i:
                    np.put(m2, [di], 0)
                    masks.append(m2)
                else:
                    d = di * 2
                    np.put(m2, [d, d+1], 0)
                    masks.append(m2)
            info[i].append(masks)

        else:
            N = info[i][1]
            m = np.ones(N, dtype=bool)
            dropN = int(N*p)
            dropInd = np.random.choice(N, dropN, replace=False)
            m[dropInd] = 0
            info[i].append(m)

        filename = args.metrics_dir + 'masks.txt'
        append_write = 'a' if os.path.exists(filename) else 'w'
        with open(filename, append_write) as f:
            f.write('%s\n' % json.dumps(m.tolist()))

    return info


def create_intelligent_masks_activation(model, acts, layer_names, args, p=0.5, percentage=True):
    info = get_dropout_layer_info(model, layer_names)
    for i in info:
        N = info[i][1]
        neuron_val = acts
        # neuron_val = np.array([np.sum(x) for x in temp])
        if(percentage):
            m = np.ones(len(neuron_val), dtype=bool)
            N = int(len(neuron_val)*p)
            rind = neuron_val.argsort()[:N]
            m[rind] = 0
        else:
            m = neuron_val > p

        info[i].append(m)

        filename = args.metrics_dir + 'masks.txt'
        append_write = 'a' if os.path.exists(filename) else 'w'
        with open(filename, append_write) as f:
            f.write('%s\n' % json.dumps(m.tolist()))

    return info



import time
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s Days ---" % ((time.time() - start_time)/86400))
