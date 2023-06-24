# hassas

## Running Experiments

To run the simmulation experiments, use the following example command:

``python main.py -model cnn --clients-per-round 10 --num-rounds 1000  --eval-every 5 -dataset cifar -lr 0.001 --num-epochs 10 --droprate 0 --batch-size 10 --seed 01938000 --optimizer feddrop --metrics-dir metrics/cifar/hassas/ -tech clt -actorgrad act``

### Parameters

The simulated experiments code supports the following parameters:

- `--optimizer`: Name of the optimizer. Choose one from: `feddrop`, `fedavg`, `fedprox`, `afd`.
- `-dataset`: Name of the dataset. Choose one from: `sent140`, `femnist`, `synthetic`, `fmnist`, `femnist_skewed`, `cifar`.
- `-model`: Name of the model.
- `--num-rounds`: Number of rounds to simulate. (Default: -1)
- `--eval-every`: Evaluate every n rounds. (Default: -1)
- `--clients-per-round`: Number of clients trained per round. (Default: -1)
- `--batch-size`: Batch size when clients train on data. (Default: 10)
- `--seed`: Seed for random client sampling and batch splitting. (Default: 0)
- `--metrics-name`: Name for metrics file. (Default: 'metrics')
- `--metrics-dir`: Directory for metrics file. (Default: 'metrics')
- `--use-val-set`: Use validation set. (Flag, no value required)
- `--droprate`: Percentage of slow devices. (Default: 0)
- `--model_droprate`: Percentage of small model. (Default: 0.5)
- `--minibatch`: None for FedAvg, else fraction. (Default: None)
- `--num-epochs`: Number of epochs when clients train on data. (Default: 2)
- `-t`: Simulation time: small, medium, or large. (Default: 'large')
- `-lr`: Learning rate for local optimizers. (Default: -1)
- `-actorgrad`: Using activations (act) or gradients (grad) to prune. (Default: None)
- `-tech`: CLT or Simple aggregation. (Default: None)
- `-pround`: Pruning at Nth round only. (Default: 10)
- `-persist`: CLT or Simple aggregation. (Default: False)
- `-save_model`: Save model? (Default: False)
