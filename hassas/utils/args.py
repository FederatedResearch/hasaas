import argparse

OPTIMIZERS = ['feddrop', 'fedavg', 'fedprox', 'afd']
DATASETS = ['sent140', 'femnist', 'femnist_large', 'shakespeare', 'celeba', 'synthetic', 'reddit', 'mnist', 'fmnist', 'femnist_skewed', 'cifar']
SIM_TIMES = ['small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        required=True)
    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--metrics-name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--use-val-set', 
                    help='use validation set;', 
                    action='store_true')
    parser.add_argument('--droprate', 
                    help='percentage of slow devices', 
                    type=float,
                    default=0)
    parser.add_argument('--model_droprate', 
                    help='percentage of small model', 
                    type=float,
                    default=0.5)

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=2)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)
    parser.add_argument('-actorgrad',
                    help='using activations (act) or gradients (grad) to prune;',
                    type=str,
                    default=None,
                    required=False)
    parser.add_argument('-tech',
                    help='CLT or Simple aggregation',
                    type=str,
                    default=None,
                    required=False)
    parser.add_argument('-pround',
                    help='pruning at Nth round only',
                    type=int,
                    default=10,
                    required=False)
    parser.add_argument('-persist',
                    help='CLT or Simple aggregation',
                    type=bool,
                    default=False,
                    required=False)
    parser.add_argument('-save_model',
                    help='save model?',
                    type=bool,
                    default=False,
                    required=False)
    return parser.parse_args()
