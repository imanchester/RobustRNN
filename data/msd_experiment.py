import numpy as np
import torch
import scipy.io as io
import os

import matplotlib.pyplot as plt
import opt.stochastic_nlsdp as nlsdp
# import models.diRNN as diRNN
import models.ee_dirnn_test as ee_dirnn
import models.lstm as lstm

import data.n_linked_msd as msd

import train_ee as train
import data.load_data as load_data

import multiprocessing
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.DoubleTensor)
multiprocessing.set_start_method('spawn', True)


# Returns the results of running model on the data in loader.
def test(model, loader):
    model.eval()

    length = loader.__len__()
    inputs = np.zeros((length,), dtype=np.object)
    outputs = np.zeros((length,), dtype=np.object)
    measured = np.zeros((length,), dtype=np.object)

    SE = np.zeros((length, model.ny))
    NSE = np.zeros((length, model.ny))

    with torch.no_grad():
        for idx, (u, y) in enumerate(loader):
            yest = model(u)
            inputs[idx] = u.numpy()
            outputs[idx] = yest.numpy()
            measured[idx] = y.numpy()

            error = yest[0].numpy() - y[0].numpy()
            mu = y[0].mean(1).numpy()
            N = error.shape[1]
            norm_factor = ((y[0].numpy() - mu[0, None])**2).sum(1)

            SE[idx] = (error ** 2 / N).sum(1) ** (0.5)
            NSE[idx] = ((error ** 2).sum(1) / norm_factor) ** (0.5)

    res = {"inputs": inputs, "outputs": outputs, "measured": measured, "SE": SE, "NSE": NSE}
    return res


def test_and_save_model(name, model, train_loader, val_loader, test_loader, log, params=None):

    nx = model.nx
    layers = model.layers
    path = "./experimental_results/gait_prediction/adversarial_attacks/w{}_l{}/".format(nx, layers)
    file_name = name + '.mat'

    train_stats = test(model, train_loader)
    val_stats = test(model, val_loader)
    test_stats = test(model, test_loader)

    data = {"validation": val_stats, "training": train_stats, "test": test_stats, "nx": model.nx, "nu": model.nu, "ny": model.ny,
            "layers": model.layers, "training_log": log}

    if params is not None:
        data = {**data, **params}

    # Create target Directory if doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " Created ")

    io.savemat(path + file_name, data)
    torch.save(model.state_dict(), path + name + ".params")

if __name__ == '__main__':

    # Number of carts
    N = 2
    bs = 500
    resolution = 15

    mu = 0.05
    eps = 1E-3

    init_var = 1.0
    init_offset = 0.05  # a small perturbatiuon to ensure strict feasbility of initial point

    max_epochs = 1000
    patience = 20

    layers = 2
    trial_number = 1
    # layers = int(sys.argv[1])
    # subject = int(sys.argv[2])

    lr_decay = 0.99
    lr = 1E-4

    print("Training models with {} layers".format(layers))
    width = 20


    # Load the data set
    sim = msd.msd_chain(N=N, T=5000, u_sd=2.0, period=100, Ts=0.05, batchsize=bs)
    # train_loader = sim.grid_ss(res=resolution)
    train_loader = sim.sim_ee(T=50000)

    val_loader = sim.simulate()
    test_loader = sim.simulate()

    nu = 1
    nx = 2 * N

    # Options for the solver
    solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=lr, mu0=100, lr_decay=lr_decay, patience=patience)

    for gamma in [0.5, 1.0, 1.5, 2.5, 10, 50, 100, 500]:
        gamma = 10
        # Train l2 gain bounded implicit model ------------------------------------------------------------------------
        name = "dl2_gamma{:1.1f}_trial{:d}".format(gamma, trial_number)
        model = ee_dirnn.ee_dirnn(nu, nx, width, layers, nl=torch.tanh)

        # Add 0.1 to ensure we are strictly feasible after initialization
        model.seb_lmi_init(epsilon=eps + init_offset, gamma=gamma, init_var=init_var, custom_seed=trial_number)

        log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options, LMIs=model.seb_lmi(gamma=gamma, epsilon=eps))

        log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                        options=solver_options)

        log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options)





        test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log, params=scaling_factors)

    log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                    options=solver_options)

    # Train Contracting model
    name = "contracting_sub{:d}_val{:d}".format(subject, val_set)
    model = diRNN.diRNN(nu, width, ny, layers, nBatches=9, nl=torch.tanh)
    model.init_l2(mu=mu, epsilon=eps+init_offset, init_var=init_var, custom_seed=this_seed+val_set)

    log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                            options=solver_options, LMIs=model.contraction_lmi(mu=mu, epsilon=eps))

    test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log, params=scaling_factors)
    run_fgsa(name, best_model, train_loader, val_loader, test_loader)

    # Train an LSTM network
    name = "LSTM_sub{:d}_val{:d}".format(subject, val_set)
    model = lstm.lstm(nu, lstm_width, ny, layers=layers)
    log, best_model = train.train_model_ipm(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                            options=solver_options)

    test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log, params=scaling_factors)
    run_fgsa(name, best_model, train_loader, val_loader, test_loader)
