import torch
import numpy as np
import multiprocessing
import scipy.io as io
import os

import models.rnn as rnn
import models.lstm as lstm
import models.RobustRnn as RobustRnn
import models.ciRNN as ciRNN
import models.dnb as dnb

import data.n_linked_msd as msd


import torch.nn.utils.clip_grad as clip_grad


torch.set_default_dtype(torch.float64)
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
    sim = msd.msd_chain(N=N, T=5000, u_sd=3.0, period=100, Ts=0.5, batchsize=20)
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


if __name__ == "__main__":

    training_batches = 100

    nu = 1
    ny = 1
    width = 10
    batches = training_batches

    path = './results/msd/'

    if not os.path.exists(path + 'lip/'):
        os.mkdir(path + 'lip/')
        print("Directory ", path, "lip/ Created")

    def run_tests(model, name):
        print("Checking Lipschitz bound of model", name)
        res = estimate_Lipschitz_bound(model, 1E-2)
        io.savemat(path + 'lip/' + "lip_" + name + ".mat", res)


    # Test Robust RNNs
    name = 'iqc-rnn_w10_gamma0.0_n4'
    model = RobustRnn.RobustRnn(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma3.0_n4'
    model = RobustRnn.RobustRnn(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma6.0_n4'
    model = RobustRnn.RobustRnn(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    name = 'iqc-rnn_w10_gamma8.0_n4'
    model = RobustRnn.RobustRnn(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

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

    # cirnn
    name = 'cirnn_w10_gamma0.0_n4'
    model = ciRNN.ciRNN(nu, width, ny, 1, nBatches=100)
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)

    # srnn
    name = 'dnb_w10_gamma0.0_n4'
    model = dnb.dnbRNN(nu, width, ny, layers=1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    run_tests(model, name)


