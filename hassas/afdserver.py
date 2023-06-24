import numpy as np

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class AFDServer:
    
    def __init__(self, client_model_high, client_model_low, masks):
        self.client_model_high = client_model_high
        self.model_high = client_model_high.get_params()
        self.selected_clients = []
        self.updates = []
        self.client_model_low = client_model_low
        self.model_low = client_model_low.get_params()
        self.masks = masks

    def select_clients(self, my_round, possible_clients, num_clients=20, droprate=0.0):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        possible_clients_high = possible_clients[0:len(possible_clients) - int(len(possible_clients)*droprate)]
        possible_clients_low = possible_clients[len(possible_clients) - int(len(possible_clients)*droprate):]

        num_clients_high = num_clients - int(droprate * num_clients)
        np.random.seed(my_round)  # make sure for each comparison, we are selecting the same clients each round
        selected_clients_high = np.random.choice(possible_clients_high, num_clients_high, replace=False)
        
        num_clients_low = int(num_clients - num_clients_high) 
        np.random.seed(my_round)  # make sure for each comparison, we are selecting the same clients each round
        selected_clients_low = np.random.choice(possible_clients_low, num_clients_low, replace=False)

        self.selected_clients = []
        for i in range(0, len(selected_clients_high)):
            self.selected_clients.append(selected_clients_high[i])
        for i in range(0, len(selected_clients_low)):
            self.selected_clients.append(selected_clients_low[i])

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, lr=-1, actorgrad=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            if c.type == 'H':
                c.model.set_params(self.model_low)
                comp, num_samples, update = c.train(num_epochs, batch_size, minibatch, actorgrad)

                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                update = self.convert_weights_to_org(update, self.masks)
                self.updates.append((num_samples, np.array(update, dtype="object"), c.type))
            elif c.type == 'L':
                c.model.set_params(self.model_low)
                comp, num_samples, update = c.train(num_epochs, batch_size, minibatch, actorgrad)

                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                update = self.convert_weights_to_org(update, self.masks)
                self.updates.append((num_samples, np.array(update, dtype="object"), c.type))
        return sys_metrics


    def aggregate_conv_dense_layer(self, weights_all, weights_high, layer_no, mask):
        weights_all[layer_no] = np.where(mask, weights_all[layer_no], weights_high[layer_no])
        weights_all[layer_no + 1] = np.where(mask, weights_all[layer_no + 1], weights_high[layer_no + 1])
        mask_expand = np.expand_dims(mask, axis=1)
        weights_all[layer_no + 2] = np.where(mask_expand, weights_all[layer_no + 2], weights_high[layer_no + 2])
        return weights_all

    def aggregate_conv_batch(self, weights_all, weights_high, layer_no, mask):
        weights_all[layer_no] = np.where(mask, weights_all[layer_no], weights_high[layer_no])
        weights_all[layer_no + 1] = np.where(mask, weights_all[layer_no + 1], weights_high[layer_no + 1])
        weights_all[layer_no + 2] = np.where(mask, weights_all[layer_no + 2], weights_high[layer_no + 2])
        weights_all[layer_no + 3] = np.where(mask, weights_all[layer_no + 3], weights_high[layer_no + 3])
        mask_expand = np.expand_dims(mask, axis=1)
        weights_all[layer_no + 4] = np.where(mask_expand, weights_all[layer_no + 4], weights_high[layer_no + 4])
        return weights_all


    def aggregate_rnn_layer(self, weights_all, weights_high, layer_no, mask, cell_0=False):
        if cell_0:
            mask0_expand = np.expand_dims(mask[0], axis=1)
            mask1 = mask[1]
            mask2_expand = np.expand_dims(mask[2], axis=1)
            weights_all[layer_no] = np.where(mask0_expand, weights_all[layer_no], weights_high[layer_no])
        else:
            mask1 = mask[0]
            mask2_expand = np.expand_dims(mask[1], axis=1)

        weights_all[layer_no] = np.where(mask1, weights_all[layer_no], weights_high[layer_no])
        weights_all[layer_no + 1] = np.where(mask1, weights_all[layer_no + 1], weights_high[layer_no + 1])    
        weights_all[layer_no + 2] = np.where(mask2_expand, weights_all[layer_no + 2], weights_high[layer_no + 2])
        return weights_all

    def update_model(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model, client_type) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln_all = [v / total_weight for v in base]
        averaged_soln_high = self.model_high
        
        for key, value in self.masks.items():
            layer_no = value[0]
            mask = value[2]
            if "dense" in key or "conv1" in key:
                averaged_soln = self.aggregate_conv_dense_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)
            
            elif "batch" in key:
                if "conv_last_batch" in key:
                    pool_reshape = (6 , 6 , averaged_soln_all[layer_no + 3].shape[0], averaged_soln_all[layer_no + 4].shape[1])
                    averaged_soln_all[layer_no + 4] = averaged_soln_all[layer_no + 4].reshape(pool_reshape)
                    averaged_soln_high[layer_no + 4] = averaged_soln_high[layer_no + 4].reshape(pool_reshape)
                    
                    averaged_soln = self.aggregate_conv_batch(averaged_soln_all, averaged_soln_high, layer_no, mask)

                    sh = (averaged_soln_all[layer_no + 4].shape[0]*averaged_soln_all[layer_no + 4].shape[1]*averaged_soln_all[layer_no + 4].shape[2], pool_reshape[3])
                    averaged_soln_all[layer_no + 4] = averaged_soln_all[layer_no + 4].reshape(sh)
                    averaged_soln_high[layer_no + 4] = averaged_soln_high[layer_no + 4].reshape(sh)
                else:
                    averaged_soln = self.aggregate_conv_batch(averaged_soln_all, averaged_soln_high, layer_no, mask)
                
            elif "conv_last" in key:
                pshape = int(np.sqrt(averaged_soln_all[layer_no + 2].shape[0]/averaged_soln_all[layer_no+1].shape[0]))
                pool_reshape = (pshape, pshape, averaged_soln_all[layer_no + 1].shape[0], averaged_soln_all[layer_no + 2].shape[1])
                averaged_soln_all[layer_no + 2] = averaged_soln_all[layer_no + 2].reshape(pool_reshape)
                averaged_soln_high[layer_no + 2] = averaged_soln_high[layer_no + 2].reshape(pool_reshape)
                
                averaged_soln = self.aggregate_conv_dense_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)

                sh = (averaged_soln_all[layer_no + 2].shape[0]*averaged_soln_all[layer_no + 2].shape[1]*averaged_soln_all[layer_no + 2].shape[2], pool_reshape[3])
                averaged_soln_all[layer_no + 2] = averaged_soln_all[layer_no + 2].reshape(sh)
                averaged_soln_high[layer_no + 2] = averaged_soln_high[layer_no + 2].reshape(sh)
            elif "rnn" in key and "cell_0" in key:
                averaged_soln = self.aggregate_rnn_layer(averaged_soln_all, averaged_soln_high, layer_no, mask, True)
            elif "rnn" in key and "cell_1" in key:
                averaged_soln = self.aggregate_rnn_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)

        temp = averaged_soln.copy()
        self.model_high = averaged_soln
        self.updates = []
        self.model_low = self.convert_weights_to_smaller(temp, self.masks)

    def weighted_std(self, average, values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        values1 = np.array([(values[i] - average)**2 for i in range(len(values))], dtype="object")
        variance = np.average(values1, weights=weights, axis=0)
        std = np.array([np.sqrt(variance[i]) for i in range(len(variance))], dtype="object")
        return std
        

    def update_model_clt(self, round):
        total_weight = 0.
        total_weight_high = 0.
        base_high = [0] * len(self.updates[0][1])
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model, client_type) in self.updates:
            total_weight += client_samples
            if(client_type == "H"):
                total_weight_high += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
                if(client_type == "H"):
                    base_high[i] += (client_samples * v.astype(np.float64))
        averaged_soln_all = [v / total_weight for v in base]
        averaged_soln_high = [v / total_weight_high for v in base_high]
        
        updates_arr = np.array(self.updates, dtype="object")
        high_update_arr = updates_arr[np.where(updates_arr[:,2] == 'H')]
        std_all = self.weighted_std(averaged_soln_all, updates_arr[:,1], updates_arr[:,0])
        std_high = self.weighted_std(averaged_soln_high, high_update_arr[:,1], high_update_arr[:,0])


        for key, value in self.masks.items():
            layer_no = value[0]
            mask = value[2]
            if "dense" in key or "conv1" in key:
                averaged_soln = self.aggregate_conv_dense_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)
                std_soln = self.aggregate_conv_dense_layer(std_all, std_high, layer_no, mask)
            
            elif "batch" in key:
                if "conv_last_batch" in key:
                    pool_reshape = (6 , 6 , averaged_soln_all[layer_no + 3].shape[0], averaged_soln_all[layer_no + 4].shape[1])
                    averaged_soln_all[layer_no + 4] = averaged_soln_all[layer_no + 4].reshape(pool_reshape)
                    averaged_soln_high[layer_no + 4] = averaged_soln_high[layer_no + 4].reshape(pool_reshape)
                    
                    averaged_soln = self.aggregate_conv_batch(averaged_soln_all, averaged_soln_high, layer_no, mask)

                    sh = (averaged_soln_all[layer_no + 4].shape[0]*averaged_soln_all[layer_no + 4].shape[1]*averaged_soln_all[layer_no + 4].shape[2], pool_reshape[3])
                    averaged_soln_all[layer_no + 4] = averaged_soln_all[layer_no + 4].reshape(sh)
                    averaged_soln_high[layer_no + 4] = averaged_soln_high[layer_no + 4].reshape(sh)

                    std_all[layer_no + 4] = std_all[layer_no + 4].reshape(pool_reshape)
                    std_high[layer_no + 4] = std_high[layer_no + 4].reshape(pool_reshape)                
                    std_soln = self.aggregate_conv_batch(std_all, std_high, layer_no, mask)
                    std_all[layer_no + 4] = std_all[layer_no + 4].reshape(sh)
                    std_high[layer_no + 4] = std_high[layer_no + 4].reshape(sh)

                else:
                    averaged_soln = self.aggregate_conv_batch(averaged_soln_all, averaged_soln_high, layer_no, mask)
                    std_soln = self.aggregate_conv_batch(std_all, std_high, layer_no, mask)

            elif "conv_last" in key:
                pshape = int(np.sqrt(averaged_soln_all[layer_no + 2].shape[0]/averaged_soln_all[layer_no+1].shape[0]))
                pool_reshape = (pshape, pshape, averaged_soln_all[layer_no + 1].shape[0], averaged_soln_all[layer_no + 2].shape[1])
                averaged_soln_all[layer_no + 2] = averaged_soln_all[layer_no + 2].reshape(pool_reshape)
                averaged_soln_high[layer_no + 2] = averaged_soln_high[layer_no + 2].reshape(pool_reshape)
                
                averaged_soln = self.aggregate_conv_dense_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)

                sh = (averaged_soln_all[layer_no + 2].shape[0]*averaged_soln_all[layer_no + 2].shape[1]*averaged_soln_all[layer_no + 2].shape[2], pool_reshape[3])
                averaged_soln_all[layer_no + 2] = averaged_soln_all[layer_no + 2].reshape(sh)
                averaged_soln_high[layer_no + 2] = averaged_soln_high[layer_no + 2].reshape(sh)
                
                #STDEV
                std_all[layer_no + 2] = std_all[layer_no + 2].reshape(pool_reshape)
                std_high[layer_no + 2] = std_high[layer_no + 2].reshape(pool_reshape)                
                std_soln = self.aggregate_conv_dense_layer(std_all, std_high, layer_no, mask)
                std_all[layer_no + 2] = std_all[layer_no + 2].reshape(sh)
                std_high[layer_no + 2] = std_high[layer_no + 2].reshape(sh)

            elif "rnn" in key and "cell_0" in key:
                averaged_soln = self.aggregate_rnn_layer(averaged_soln_all, averaged_soln_high, layer_no, mask, True)
                std_soln = self.aggregate_rnn_layer(std_all, std_high, layer_no, mask, True)
            elif "rnn" in key and "cell_1" in key:
                averaged_soln = self.aggregate_rnn_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)
                std_soln = self.aggregate_rnn_layer(std_all, std_high, layer_no, mask)

        new_params = [np.random.normal(averaged_soln[i], (std_soln[i]/np.sqrt(round)), averaged_soln[i].shape) for i in range(len(averaged_soln))]
        self.model_high = new_params
        temp = new_params.copy()
        self.model_low = self.convert_weights_to_smaller(temp, self.masks)
        self.updates = []
        return new_params

    def test_client_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            if client.type == 'H':
                c_metrics = client.test(set_to_use)
                metrics[client.id] = c_metrics
            elif client.type == 'L':
                c_metrics = client.test(set_to_use)
                metrics[client.id] = c_metrics
        return metrics

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            if client.type == 'H':
                client.model.set_params(self.model_low)
                c_metrics = client.test(set_to_use)
                metrics[client.id] = c_metrics
            elif client.type == 'L':
                client.model.set_params(self.model_low)
                c_metrics = client.test(set_to_use)
                metrics[client.id] = c_metrics
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        types = {c.id: c.type for c in clients}
        return ids, groups, num_samples, types

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model_high.set_params(self.model_low)
        self.client_model_low.set_params(self.model_low)
        model_sess_high =  self.client_model_high.sess
        model_sess_low =  self.client_model_low.sess
        return (self.client_model_high.saver.save(model_sess_high, path+"_large.ckpt"), self.client_model_low.saver.save(model_sess_low, path+"_small.ckpt"))


    def load_model(self, path):
        with self.client_model_high.sess as sess:
            self.client_model_high.saver.restore(sess, path+"_large.ckpt")
        with self.client_model_low.sess as sess:
            self.client_model_low.saver.restore(sess, path+"_small.ckpt")

    def close_model(self):
        self.client_model_high.close()
        self.client_model_low.close()

    def updated_cnn_mask(self):
        weights = self.model_high
        info = self.masks
        for layer in info:
            if "conv" in layer:
                mask = info[layer][2]
                layer_no = info[layer][0]
                total_filters = info[layer][1]
                filter_val = []
                for i in range(total_filters):
                    temp = weights[layer_no][:,:,:,i]
                    norm = np.linalg.norm(temp.flatten(), ord=1)
                    filter_val.append(norm)
                
                m = np.ones(len(filter_val), dtype=bool)
                dropN = total_filters - np.count_nonzero(mask)
                rind = np.array(filter_val).argsort()[:dropN]
                m[rind] = 0
                self.masks[layer][2] = m
        return self.masks

    def dropout(self, weights, layer_no, mask, bias=True, layer = ''):
        if 'conv' in layer and 'batch' in layer:
            weights[layer_no] = weights[layer_no][:,:,:,mask]
            if(bias): 
                weights[layer_no + 1] = weights[layer_no + 1][mask]
                weights[layer_no + 2] = weights[layer_no + 2][mask]
                weights[layer_no + 3] = weights[layer_no + 3][mask]
                if(len(weights) >= layer_no + 4):
                    weights[layer_no + 4] = weights[layer_no + 4][:,:,mask,:] 
            else:
                weights[layer_no + 3] = weights[layer_no + 3][:,:,mask,:] 
        elif 'conv' in layer:
            weights[layer_no] = weights[layer_no][:,:,:,mask]
            if (bias): 
                weights[layer_no + 1] = weights[layer_no + 1][mask]
                if (len(weights) >= layer_no + 2):
                    weights[layer_no + 2] = weights[layer_no + 2][:,:,mask,:] 
            else:
                weights[layer_no + 1] = weights[layer_no + 1][:,:,mask,:] 

        elif 'rnn' in layer:
            if 'cell_0' in layer:
                mask1 = mask[1]
                mask2 = mask[2]
                weights[layer_no] = weights[layer_no][mask[0],:]
                weights[layer_no] = weights[layer_no][:,mask1]

            else:
                mask1 = mask[0]
                mask2 = mask[1]
                weights[layer_no] = weights[layer_no][:,mask1] #for previous connections

            if (bias): #weight matrix have different shape depending on bise set true of false
                weights[layer_no + 1] = weights[layer_no + 1][mask1]
                if (len(weights) >= layer_no + 2):
                    weights[layer_no + 2] = weights[layer_no + 2][mask2, :] #for next layer connections
            else:
                weights[layer_no + 1] = weights[layer_no + 1][mask2, :] #for next layer connections

        else:
            weights[layer_no] = weights[layer_no][:,mask] #for previous connections
            if (bias): #weight matrix have different shape depending on bise set true of false
                weights[layer_no + 1] = weights[layer_no + 1][mask]
                if (len(weights) >= layer_no + 2):
                    weights[layer_no + 2] = weights[layer_no + 2][mask, :] #for next layer connections
            else:
                weights[layer_no + 1] = weights[layer_no + 1][mask, :] #for next layer connections
        return weights


    def update_to_org(self, weights, layer_no, mask, layer = ''):
        prev_con = weights[layer_no]
        prev_bias = weights[layer_no + 1]
        next_con = weights[layer_no + 2]
        removed_neurons_indexes = np.where(mask == False)[0]
        if 'rnn' in layer:
            if 'cell_0' in layer:
                removed_neurons_indexes = np.where(mask[0] == False)[0]
                for i in removed_neurons_indexes:
                    prev_con = np.insert(prev_con, i, 0, axis=0)

                removed_neurons_indexes = np.where(mask[1] == False)[0]
                for i in removed_neurons_indexes:
                    prev_con = np.insert(prev_con, i, 0, axis=1)
                    prev_bias = np.insert(prev_bias, i, 0, axis=0)

                removed_neurons_indexes = np.where(mask[2] == False)[0]
                for i in removed_neurons_indexes:
                    next_con = np.insert(next_con, i, 0, axis=0)
            else:
                removed_neurons_indexes = np.where(mask[0] == False)[0]
                for i in removed_neurons_indexes:
                    prev_con = np.insert(prev_con, i, 0, axis=1)
                    prev_bias = np.insert(prev_bias, i, 0, axis=0)

                removed_neurons_indexes = np.where(mask[1] == False)[0]
                for i in removed_neurons_indexes:
                    next_con = np.insert(next_con, i, 0, axis=0)

            weights[layer_no] = prev_con
            weights[layer_no + 1] = prev_bias
            weights[layer_no + 2] = next_con

        elif 'batch' in layer:
            prev_con = weights[layer_no]
            prev_bias = weights[layer_no + 1]
            batch1 = weights[layer_no + 2]
            batch2 = weights[layer_no + 3]
            next_con = weights[layer_no + 4]
            removed_neurons_indexes = np.where(mask == False)[0]     
            for i in removed_neurons_indexes:
                prev_con = np.insert(prev_con, i, 0, axis=3)
                prev_bias = np.insert(prev_bias, i, 0, axis=0)
                batch1 = np.insert(batch1, i, 0, axis=0)
                batch2 = np.insert(batch2, i, 0, axis=0)
                next_con = np.insert(next_con, i, 0, axis=2)
        
            weights[layer_no] = prev_con
            weights[layer_no + 1] = prev_bias
            weights[layer_no + 2] = batch1
            weights[layer_no + 3] = batch2
            weights[layer_no + 4] = next_con

        else:
            prev_con = weights[layer_no]
            prev_bias = weights[layer_no + 1]
            next_con = weights[layer_no + 2]
            removed_neurons_indexes = np.where(mask == False)[0]
            if 'conv' in layer:
                for i in removed_neurons_indexes:
                    prev_con = np.insert(prev_con, i, 0, axis=3)
                    prev_bias = np.insert(prev_bias, i, 0, axis=0)
                    next_con = np.insert(next_con, i, 0, axis=2)
            else:
                for i in removed_neurons_indexes:
                    prev_con = np.insert(prev_con, i, 0, axis=1)
                    prev_bias = np.insert(prev_bias, i, 0, axis=0)
                    next_con = np.insert(next_con, i, 0, axis=0)
                    
            weights[layer_no] = prev_con
            weights[layer_no + 1] = prev_bias
            weights[layer_no + 2] = next_con
        
        return weights

    def update_to_org_activations(self, weights, mask):
        prev_con = weights
        removed_neurons_indexes = np.where(mask == False)[0]
        for i in removed_neurons_indexes:
            prev_con = np.insert(prev_con, i, 0, axis=0)            
        weights = prev_con
        return weights

    def convert_weights_to_org(self, weights, info):
        bias = True
        for layer in info:
            mask = info[layer][2]
            layer_no = info[layer][0]
            if "conv_last_batch" in layer:
                pool_reshape = (6 , 6 , weights[layer_no + 3].shape[0], weights[layer_no + 4].shape[1])
                weights[layer_no + 4] = weights[layer_no + 4].reshape(pool_reshape)
                weights = self.update_to_org(weights, layer_no, mask, layer)
                sh = (weights[layer_no + 4].shape[0]*weights[layer_no + 4].shape[1]*weights[layer_no + 4].shape[2], pool_reshape[3])
                weights[layer_no + 4] = weights[layer_no + 4].reshape(sh)

            elif "conv_last" in layer:
                pshape = int(np.sqrt(weights[layer_no + 2].shape[0]/weights[layer_no+1].shape[0]))
                pool_reshape = (pshape, pshape, weights[layer_no + 1].shape[0], weights[layer_no + 2].shape[1])
                weights[layer_no + 2] = weights[layer_no + 2].reshape(pool_reshape)
                weights = self.update_to_org(weights, layer_no, mask, layer)
                sh = (weights[layer_no + 2].shape[0]*weights[layer_no + 2].shape[1]*weights[layer_no + 2].shape[2], pool_reshape[3])
                weights[layer_no + 2] = weights[layer_no + 2].reshape(sh)
            elif "conv" in layer:
                weights = self.update_to_org(weights, layer_no, mask, layer)
            else:
                weights = self.update_to_org(weights, layer_no, mask, layer)
            
        return weights

    def convert_weights_to_smaller(self, weights, info):
        bias = True
        for layer in info:
            mask = info[layer][2]
            layer_no = info[layer][0]
            if "conv_last_batch" in layer:
                pool_reshape = (6 , 6 , weights[layer_no + 3].shape[0], weights[layer_no + 4].shape[1])
                weights[layer_no + 4] = weights[layer_no + 4].reshape(pool_reshape)
                weights = self.dropout(weights, layer_no, mask, bias, layer)
                sh = (weights[layer_no + 4].shape[0]*weights[layer_no + 4].shape[1]*weights[layer_no + 4].shape[2], pool_reshape[3])
                weights[layer_no + 4] = weights[layer_no + 4].reshape(sh)

            elif "conv_last" in layer:
                pshape = int(np.sqrt(weights[layer_no + 2].shape[0]/weights[layer_no+1].shape[0]))
                pool_reshape = (pshape, pshape, weights[layer_no + 1].shape[0], weights[layer_no + 2].shape[1])
                weights[layer_no + 2] = weights[layer_no + 2].reshape(pool_reshape)
                weights = self.dropout(weights, layer_no, mask, bias, layer)
                sh = (weights[layer_no + 2].shape[0]*weights[layer_no + 2].shape[1]*weights[layer_no + 2].shape[2], pool_reshape[3])
                weights[layer_no + 2] = weights[layer_no + 2].reshape(sh)
            elif "conv" in layer:
                weights = self.dropout(weights, layer_no, mask, bias, layer)
            elif "rnn" in layer:
                weights = self.dropout(weights, layer_no, mask, bias, layer)
            else:
                weights = self.dropout(weights, layer_no, mask, bias)
        return weights
