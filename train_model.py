import os
import argparse
import torch
import multiprocessing
import numpy as np
import scipy.io as io

# Import various models and data sets
import data.n_linked_msd as msd
import data.load_tac_data as data
import opt.snlsdp_ipm as ipm
import opt.train as train
import models.ciRNN as ciRNN
import models.lstm as lstm
import models.rnn as rnn
import models.RobustRnn as RobustRnn
import models.dnb as dnb


parser = argparse.ArgumentParser(description='Training Robust RNNs')

parser.add_argument('--model', type=str, default='RobustRnn',
                    choices=['lstm', 'rnn', 'cirnn', 'RobustRnn', 'dnb'],
                    help='Select model type')

parser.add_argument('--multiplier', type=str, default='Neuron',
                    choices=['Network', 'Neuron', 'Layer'],
                    help='The set of multipliers to use. Network is most expressive, least scalable\
                        (It also doesn\'t work...). \
                          then neuron then layer.')

parser.add_argument('--supply_rate', type=str, default='stable',
                    choices=['dl2_gain', 'stable'],
                    help='Supply rate to be used. dl2_gain is a differential \
                          l2 gain bound with gain specified by gamma.\
                          Stable means there is no supply rate.')

parser.add_argument('--gamma', type=float, default=5.0,
                    help='L2 gain bound for l2rnn and iqc-rnn\'s')

parser.add_argument('--gamma_var', type=bool, default=False,
                    help='Treat gamma as a decision variable with a penalty')

parser.add_argument('--width', type=int, default=10,
                    help='size of state space in model')

parser.add_argument('--res_size', type=int, default=10,
                    help='width of hidden layers in model')

parser.add_argument('--init_type', type=str, default='n4sid',
                    choices=['n4sid', 'A', 'B'])

parser.add_argument('--init_var', type=float, default=1.0,
                    help='Initial variance of the state transition matrices')

parser.add_argument('--depth', type=int, default=1,
                    help='number of hidden layers in model.\
                          iqc-rnn currently only accepts 1.')

# Solver options
parser.add_argument('--method', type=str, default='ipm',
                    choices=['ipm', 'penalty', 'pgd', 'sca'],
                    help='Method of constrainted optimization.\
                          Currently only ipm is working.')

parser.add_argument('--lr', type=float, default=1E-3,
                    help='Learning Rate')

parser.add_argument('--lr_decay', type=float, default=0.25,
                    help='Value in (0,1] specifying exponential\
                          decay rate.')

parser.add_argument('--patience', type=int, default=10,
                    help='Number of epochs without validation\
                          improvement before finishing')

parser.add_argument('--max_epochs', type=int, default=2000,
                    help='Maximum number of epochs.')

parser.add_argument('--name', type=str, default='test',
                    help='name of the trial')

parser.add_argument('--dataset', type=str, default='msd',
                    choices=['Tac2017'],
                    help='Data set to test on')

parser.add_argument('--N', type=int, default=4,
                    help='Number of carraiges for msd.')

parser.add_argument('--save', type=bool, default=True,
                    help='Save results?')

# Parameters for ipm
parser.add_argument('--mu0', type=float, default=100.0,
                    help='Initial value of barrier paramers')


parser.add_argument('--mu_rate', type=float, default=10.0,
                    help='Rate of increase in barrier weight.')


parser.add_argument('--mu_max', type=float, default=1E6,
                    help='Maximum weight on barriere parameter.')

parser.add_argument('--clip_at', type=float, default=200.0,
                    help='Clip gradient at')

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed to use for both numpy and torch')

args = parser.parse_args()

# Other stuff
torch.set_default_dtype(torch.float64)  # double precision for SDP
torch.set_printoptions(precision=4)
multiprocessing.set_start_method('spawn', True)

# Set a random seed if provided
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# Returns the performance metics for running model on loader


def test(model, loader):
    model.eval()

    # This is a pretty dodgy way of doing this.
    length = loader.__len__()
    inputs = np.zeros((length,), dtype=np.object)
    outputs = np.zeros((length,), dtype=np.object)
    measured = np.zeros((length,), dtype=np.object)

    SE = np.zeros((length, model.ny))
    NSE = np.zeros((length, model.ny))

    with torch.no_grad():
        for (idx, u, y) in loader:

            yest = model(u)
            inputs[idx] = np.split(u.numpy(), u.shape[0], 0)
            outputs[idx] = np.split(yest.numpy(), yest.shape[0], 0)
            measured[idx] = np.split(y.numpy(), y.shape[0], 0)

            error = yest[0].numpy() - y[0].numpy()
            mu = y[0].mean(1).numpy()
            N = error.shape[1]
            norm_factor = ((y[0].numpy() - mu[0, None])**2).sum(1)

            SE[idx] = (error ** 2 / N).sum(1) ** (0.5)
            NSE[idx] = ((error ** 2).sum(1) / norm_factor) ** (0.5)

    res = {"inputs": inputs, "outputs": outputs,
           "measured": measured, "SE": SE, "NSE": NSE}
    return res


def test_and_save_model(path, name, model, train_loader,
                        val_loader, test_loader, log, params=None):

    print("Testing and Saving Models")

    # Create target Directory if doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " Created")

    file_name = '/' + name + '.mat'

    # Test performance and store in dict for saving
    train_stats = test(model, train_loader)
    val_stats = test(model, val_loader)
    test_stats = test(model, test_loader)

    data = {"validation": val_stats, "training": train_stats,
            "test": test_stats, "nx": model.nx, "nu": model.nu,
            "ny": model.ny, "training_log": log}

    if params is not None:
        data = {**data, **params}

    # Save
    io.savemat(path + file_name, data)
    torch.save(model.state_dict(), path + '/' + name + ".params")


def generate_model(nu, ny, batches, args, loader=None, solver="SCS"):
    r'Function to easily re-generate models for training on different data sets.'

    print('Creating model', args.model, ' width = ', args.width)
    if args.model == "cirnn":
        model = ciRNN.ciRNN(nu, args.width, ny, args.depth, nBatches=batches)
        model.init_l2(0.0, 1E-3)
        constraints = {"lmi": model.contraction_lmi(0, 1E-5),
                       "inequality": None}

    elif args.model == "RobustRnn":
        model = RobustRnn.RobustRnn(nu, args.width, ny, args.res_size, nBatches=batches,
                                    method=args.multiplier, supply_rate=args.supply_rate)

        if args.supply_rate == "dl2_gain":
            print('\t supply rate: dl2 gamma = ', args.gamma)

            if args.init_type == 'n4sid':
                model.init_lipschitz_ss(
                    gamma=args.gamma, loader=loader, solver=solver)
            else:
                model.initialize_lipschitz_LMI(
                    gamma=args.gamma, eps=1E-3, init_var=args.init_var)

            constraints = {"lmi": model.lipschitz_LMI(gamma=args.gamma, eps=1E-5),
                           "inequality": [model.multiplier_barrier]}

        elif args.supply_rate == "stable":
            print('\t supply rate: stable')
            if args.init_type == 'n4sid':
                model.init_stable_ss(loader)
            else:
                model.initialize_stable_LMI(
                    eps=1E-3, init_var=args.init_var, obj=args.init_type)
            constraints = {"lmi": model.stable_LMI(eps=1E-5),
                           "inequality": [model.multiplier_barrier]}

    elif args.model == "rnn":
        model = rnn.rnn(nu, args.width, ny, nBatches=batches)
        constraints = {"lmi": None,
                       "inequality": None}

    elif args.model == "lstm":  # only constraint is invertible E
        model = lstm.lstm(nu, args.width, ny, nBatches=batches)
        constraints = {"lmi": None,
                       "inequality": None}

    elif args.model == 'dnb':
        model = dnb.dnbRNN(nu, args.width, ny, nBatches=batches)
        constraints = {"lmi": model.norm_ball_lmi(eps=0.001),
                       "inequality": None}

    return model, constraints


if __name__ == "__main__":

    # Load solver options
    solver_options = ipm.make_default_options(max_epochs=args.max_epochs, lr=args.lr, clip_at=args.clip_at,
                                              mu0=args.mu0, mu_rate=args.mu_rate, mu_max=args.mu_max,
                                              lr_decay=args.lr_decay, patience=args.patience)

    print("Running model on dataset msd")
    # N = args.N
    # sim = msd.msd_chain(N=N, T=5000, u_sd=3.0,
    #                     period=100, Ts=0.5, batchsize=20)

    # Load previously simulated msd data
    # loaders, lin_loader = msd.load_saved_data()
    loaders = data.load(1)

    train_seq_len = 1000
    training_batches = 100
    mini_batch_size = 1

    nu = 1
    ny = 1

    # The training loader is used to initalize the model using n4sid
    model, Con = generate_model(
        nu, ny, training_batches, args, loader=loaders["Training"])

    log, best_model = train.train_model(
        model, loaders=loaders, method="ipm", options=solver_options, constraints=Con, mse_type='mean')

    if args.save:
        path = './results/msd'
        name = args.model + '_w' + \
            str(args.width) + '_gamma' + str(args.gamma) + '_n' + str(args.N)
        test_and_save_model(path, name, model, loaders["Training"],
                            loaders["Validation"], loaders["Test"], log)
