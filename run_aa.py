import torch
import numpy as np
import data.load_data as load_data
from scipy.optimize import minimize, NonlinearConstraint
import multiprocessing
import scipy.io as io
import argparse
import os

import models.rnn as rnn
import models.lstm as lstm
import models.iqcRnn_reservoir as iqcRNN
import models.diRNN as diRNN
import models.dnb as dnb
import sys

import data.n_linked_msd as msd

from os import listdir
from os.path import isfile, join

import torch.nn.utils.clip_grad as clip_grad
import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float64)  # Need double precision because semidefinite programming is hard
torch.set_printoptions(precision=4)
multiprocessing.set_start_method('spawn', True)


# Numerically try and maximize the Lipschitz constant of the model. 
def estimate_Lipschitz_bound(model, alpha):
    tol_change = 1E-3

    model.eval()

    # Initialize signals
    u = torch.zeros((1, 1, 500), requires_grad=True)
    du = torch.randn((1, 1, 500), requires_grad=True)
    du.data = du.data * 0.05

    optimizer = torch.optim.Adam([u, du], lr=alpha)

    J_best = 0
    no_decrease_counter = 0

    for ii in range(2000):
        # Try and maximize lipschitz constant
        def closure():
            optimizer.zero_grad()
            yest1 = model(u)
            yest2 = model(u + du)
            J = -((yest2 - yest1).norm()**2) / (du.norm()**2)
            J.backward()
            clip_grad.clip_grad_norm_([u, du], 100, 2)
            return J
        J = optimizer.step(closure)

        # Check for sufficient improvement in objective
        if J < J_best - tol_change:
            J_best = J.item()
            no_decrease_counter = 0

        else:
            no_decrease_counter += 1

        # If objective is not decreaseing decrease learning rate
        # or finish optimization if lr is too small
        if no_decrease_counter >= 40:
            no_decrease_counter = 0

            # If min lr is reached break
            lrs = [p['lr'] for p in optimizer.param_groups]
            if lrs[0] <= 1E-5:
                break

            print('reducing learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        grad_norm = torch.sqrt(u.grad.norm() ** 2 + du.grad.norm() ** 2)
        print('iter', ii, '\tgamma = {:2.3f} \t g={:2.1f}'.format((-J).sqrt().item(), grad_norm.item()))

    # Simulate true model behaviour at the two points found.
    sim = msd.msd_chain(N=N, T=5000, u_sd=3.0, period=100, Ts=0.2, batchsize=20)
    _, Y1 = sim.simulate(u[0].detach().numpy().T)
    yest1 = model(u)

    NSE1 = (np.linalg.norm(yest1.detach().numpy() - Y1) / np.linalg.norm(Y1))

    v = (u + du)
    _, Y2 = sim.simulate(v[0].detach().numpy().T)
    yest2 = model(v)
    NSE2 = np.linalg.norm(yest2.detach().numpy() - Y2) / np.linalg.norm(Y2)

    # Calculare the Lipschitz constant
    gamma = (yest1 - yest2).norm().detach().numpy() / (du).norm().detach().numpy()

    res = {"gamma": gamma, "u1": u.detach().numpy(), "u2": v.detach().numpy(),
           "y1": yest1.detach().numpy(), "y2": yest2.detach().numpy(), "True1": Y1, "True2": Y2,
           "NSE1": NSE1, "NSE2": NSE2}

    return res


# Numerically try and find the worst case perturbation at 5% of signal size.
# Use interior point method.
def estimate_wc_pert(model, alpha, alpha_decay=0.90):
    epsilon = 0.05
    u_max = 40.0
    tol_change = 1E-2
    wash_per = 200

    model.eval()

    # Initialize signals
    u = torch.randn((1, 1, 500), requires_grad=True)
    du = torch.zeros((1, 1, 500), requires_grad=True)
    u.data = u.data * 0.05

    # Used to calculate the true model response.
    sim = msd.msd_chain(N=4, T=5000, u_sd=3.0, period=100, Ts=0.2, batchsize=20)
    J_best = 0
    no_decrease_counter = 0

    optimizer = torch.optim.SGD([u, du], lr=alpha)

    J_best = 0
    no_decrease_counter = 0

    for ii in range(2000):
        # Try and maximize lipschitz constant
        def closure():
            optimizer.zero_grad()
            washout = torch.zeros(1, 1, wash_per, requires_grad=False)
            u1 = torch.cat([washout, u + du], 2)
            u2 = torch.cat([washout, u], 2)

            yest1 = model(u1)
            yest2 = sim.torch_sim(u2)
            J = -((yest2 - yest1).norm()**2)
            J.backward()
            clip_grad.clip_grad_norm_([u, du], 2, 2)
            return J
        J = optimizer.step(closure)

        if u.norm() > u_max:
            u.data = u.data / u.data.norm() * u_max
            # print("projecting u onto norm ball")

        # project back onto norm ball if we step out
            # project back onto norm ball
        if float(du.data.norm()) / float(u.norm()) > epsilon:
            du.data = du.data * epsilon / du.data.norm() * u.norm()
            alpha *= alpha_decay
            # print("projecting du onto feasible set.")

        # Check for sufficient improvement in objective
        if J < J_best - tol_change:
            J_best = J.item()
            no_decrease_counter = 0

        else:
            no_decrease_counter += 1

        # If objective is not decreaseing decrease learning rate
        # or finish optimization if lr is too small
        if no_decrease_counter >= 20:
            no_decrease_counter = 0

            # If min lr is reached break
            lrs = [p['lr'] for p in optimizer.param_groups]
            if lrs[0] <= 1E-5:
                break

            print('reducing learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        grad_norm = torch.sqrt(u.grad.norm() ** 2 + du.grad.norm() ** 2)
        print('iter', ii, '\tgamma = {:2.3f} \t g={:2.1f}'.format((-J).sqrt().item(), grad_norm.item()))

    washout = torch.zeros(1, 1, wash_per, requires_grad=False)
    u1 = torch.cat([washout, u + du], 2)
    u2 = torch.cat([washout, u], 2)

    # Simulate true model behaviour at the two points found.
    sim = msd.msd_chain(N=N, T=5000, u_sd=3.0, period=100, Ts=0.2, batchsize=20)
    _, Y1 = sim.simulate(u1[0].detach().numpy().T)
    yest1 = model(u1)

    NSE1 = (np.linalg.norm(yest1.detach().numpy()[0, 0, wash_per:] - Y1[0, 0, wash_per:]) / np.linalg.norm(Y1[0, 0, wash_per:]))

    # Simulate with adversarial input
    _, Y2 = sim.simulate(u2[0].detach().numpy().T)
    yest2 = model(u2)
    NSE2 = (np.linalg.norm(yest2.detach().numpy()[0, 0, wash_per:] - Y2[0, 0, wash_per:]) / np.linalg.norm(Y2[0, 0, wash_per:]))

    res = {"gamma": np.sqrt(-J_best), "u1": u1.detach().numpy(), "u2": u2.detach().numpy(),
           "y1": yest1.detach().numpy(), "y2": yest2.detach().numpy(), "True1": Y1, "True2": Y2,
           "NSE1": NSE1, "NSE2": NSE2}

    return res


# Numerically try and find the worst case perturbation at 5% of signal size.
# Use interior point method.
def estimate_wc_gen(model, alpha, alpha_decay=0.90):
    u_max = 10
    tol_change = 1E-2
    wash_per = 200
    seq_len = 500
    model.eval()

    # Initialize signals
    u = torch.randn((1, 1, seq_len), requires_grad=True)
    u.data = u.data * 0.5

    # Used to calculate the true model response.
    sim = msd.msd_chain(N=4, T=5000, u_sd=3.0, period=100, Ts=0.5, batchsize=20)

    # Load the validation input
    loaders, lin_loader = msd.load_saved_data()
    # u_bar = [u for idx, u, y in loaders["Validation"]][0][:, :, 200:seq_len + 200]
    # u_bar = [u for idx, u, y in loaders["Validation"]][0]
    # u_max = 0.1 * u_bar.norm()

    J_best = 0
    no_decrease_counter = 0

    optimizer = torch.optim.SGD([u], lr=alpha)

    J_best = 0
    no_decrease_counter = 0

    for ii in range(2000):
        # Try and maximize lipschitz constant
        def closure():
            optimizer.zero_grad()
            washout = torch.zeros(1, 1, wash_per, requires_grad=False)
            # washout = u_bar[:, :, :200]
            u1 = torch.cat([washout, u], 2)

            yest1 = model(u1)
            yest2 = sim.torch_sim(u1)
            J = -((yest2 - yest1).norm()**2) / yest2.norm()**2
            J.backward()
            clip_grad.clip_grad_norm_([u], 1.0, 2)
            return J
        J = optimizer.step(closure)

        if u.norm() > u_max:
            u.data = u.data / u.data.norm() * u_max
            # print("projecting u onto norm ball")

        # Check for sufficient improvement in objective
        if J < J_best - tol_change:
            J_best = J.item()
            no_decrease_counter = 0
        else:
            no_decrease_counter += 1

        # If objective is not decreaseing decrease learning rate
        # or finish optimization if lr is too small
        if no_decrease_counter >= 5:
            no_decrease_counter = 0

            # If min lr is reached break
            lrs = [p['lr'] for p in optimizer.param_groups]
            if lrs[0] <= 1E-3:
                break

            print('reducing learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        grad_norm = u.grad.norm()
        print('iter', ii, '\tgamma = {:2.3f} \t g={:2.3f}'.format((-J).sqrt().item(), grad_norm.item()))

    # Calculate final statistics
    washout = torch.zeros(1, 1, wash_per, requires_grad=False)
    u1 = torch.cat([washout, u], 2)

    # Simulate true model behaviour at the two points found.
    sim = msd.msd_chain(N=N, T=5000, u_sd=3.0, period=100, Ts=0.2, batchsize=20)
    sim.v_sd = 0.0
    _, Y1 = sim.simulate(u1[0].detach().numpy().T)
    yest1 = model(u1)

    NSE = (np.linalg.norm(yest1.detach().numpy()[0, 0, wash_per:] - Y1[0, 0, wash_per:]) / np.linalg.norm(Y1[0, 0, wash_per:]))

    res = {"gamma": np.sqrt(-J_best), "u1": u1.detach().numpy(),
           "y1": yest1.detach().numpy(), "True1": Y1,
           "NSE": NSE}

    return res

if __name__ == "__main__":

    N = 4
    sim = msd.msd_chain(N=N, T=5000, u_sd=3.0, period=100, Ts=0.2, batchsize=20)

    data = sim.simulate()

    train_seq_len = 1000
    training_batches = 100
    mini_batch_size = 1

    nu = 1
    ny = 1
    width = 10
    batches = training_batches

    path = './results/msd/'

    if not os.path.exists(path + 'lip/'):
        os.mkdir(path + 'lip/')
        os.mkdir(path + 'wcg/')
        os.mkdir(path + 'wcp/')
        print("Directory ", path, "lip/ Created")

    def run_tests(model, name):

        print("Checking Lipschitz bound of model", name)
        res = estimate_Lipschitz_bound(model, 1E-2)
        io.savemat(path + 'lip/' + "lip_" + name + ".mat", res)

        print("Checking WCG of model", name)
        # res = estimate_wc_gen(model, 1E0)
        # io.savemat(path + 'wcg/' + "wcg_" + name + ".mat", res)

        print("Checking WCP of model", name)
        # res = estimate_wc_pert(model, 1E-2)
        # io.savemat(path + 'wcp/' + "wcp_" + name + ".mat", res)


    # name = 'iqc-rnn_w10_gamma3.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # run_tests(model, name)

    name = 'iqc-rnn_w10_gamma20.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma10.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    # iqc-rnns
    # lstm
    name = 'lstm_w10_gamma0.0_n4'
    model = lstm.lstm(nu, width, ny, layers=1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    # rnn
    name = 'rnn_w10_gamma0.0_n4'
    model = rnn.rnn(nu, width, ny, 1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma3.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma6.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    # iqc-rnns
    name = 'iqc-rnn_w10_gamma0.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    # cirnn
    name = 'cirnn_w10_gamma0.0_n4'
    model = diRNN.diRNN(nu, width, ny, 1, nBatches=100)
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    # srnn
    name = 'dnb_w10_gamma0.0_n4'
    model = dnb.dnbRNN(nu, width, ny, layers=1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    # name = 'iqc-rnn_w10_gamma3.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # run_tests(model, name)

    # name = 'iqc-rnn_w10_gamma6.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # run_tests(model, name)

    name = 'iqc-rnn_w10_gamma12.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma8.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma10.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

