import torch
import numpy as np
import data.load_data as load_data
from scipy.optimize import minimize, NonlinearConstraint
import multiprocessing
import scipy.io as io
import os

import models.rnn as rnn
import models.lstm as lstm
import models.iqcRnn_V2 as iqcRNN
import models.diRNN as diRNN
import models.dnb as dnb
import data.n_linked_msd as msd

import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float64)  # Need double precision because semidefinite programming is hard
torch.set_printoptions(precision=4)
multiprocessing.set_start_method('spawn', True)


def test_performance(model, period, sd, samples=10, sim_len=500):
    model.eval()
    sim = msd.msd_chain(N=4, T=sim_len, u_sd=sd, period=period, Ts=0.5, batchsize=samples)
    loader = sim.sim_rand_ic(sim_len, samples, mini_batch_size=samples)
    wash_per = 200

    for idx, U, Y in loader:
        with torch.no_grad():
            Yest = model(U)
            SE = (Y[:, :, wash_per:] - Yest[:, :, wash_per:]).norm(dim=(1, 2)).detach()
            NSE = ((Y[:, :, wash_per:] - Yest[:, :, wash_per:]).norm(dim=(1, 2)) / Y[:, :, wash_per:].norm(dim=(1, 2))).detach()

        sensitivity = estimate_sensitivity(model, U)

    res = {"NSE": NSE, "SE": SE, "Sensitivity": sensitivity}
    return res


def estimate_sensitivity(model, u):

    # used when calculating the jacobian. As this method of calulating the jacobian 
    def insert_model_dim(model, u):
        upad = u.unsqueeze(1)
        Y = model(upad)
        return Y[:, 0, :]

    batches = u.shape[0]
    length = u.shape[2]

    sensitivity_dist = torch.zeros(batches)
    for idx in range(batches):
        U_rep = torch.nn.Parameter(u[idx, 0].repeat(length, 1))
        Yest = insert_model_dim(model, U_rep)
        Yest.backward(torch.eye(length))

        Jac = U_rep.grad.data.detach()

        [U, S, V] = torch.svd(Jac)
        sensitivity_dist[idx] = S[0]

    return sensitivity_dist


if __name__ == "__main__":

    N = 4

    train_seq_len = 1000
    training_batches = 100
    mini_batch_size = 1

    nu = 1
    ny = 1
    width = 10
    batches = training_batches

    path = './results/msd/'

    if not os.path.exists(path + 'lip/'):
        print("Directory ", path, "lip/ Created")

    def vary_amplitude(model):
        print("\t Testing amplitude")
        samples = 5
        test_points = 5
        period = 100

        sensitivity_dist = np.zeros((test_points, samples))
        NSE_dist = np.zeros((test_points, samples))
        SE_dist = np.zeros((test_points, samples))
        amps = np.linspace(0.5, 10.5, test_points)


        for idx in range(test_points):
            res = test_performance(model, period, amps[idx], samples=samples)
            NSE_dist[idx, :] = res["NSE"]
            SE_dist[idx, :] = res["SE"]
            sensitivity_dist[idx, :] = res["Sensitivity"]

        return {"Sensitivity": sensitivity_dist, "amps": amps, "NSE": NSE_dist, "SE": SE_dist, "period": period}

    def vary_period(model):
        print("\t Testing period")
        samples = 50
        periods = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
        amp = 3
        test_points = periods.__len__()

        sensitivity_dist = np.zeros((test_points, samples))

        for (idx, period) in enumerate(periods):
            res = test_performance(model, period, amp, samples=samples)
            sensitivity_dist[idx, :] = res["Sensitivity"]

        return {"amp": amp, "Sensitivity": sensitivity_dist, "period": periods}

    def test_training_dist(model):
        print("\t Testing amplitude")
        samples = 50
        period = 100
        amp = 3

        res = test_performance(model, period, amp, samples=samples, sim_len=1000)
        NSE_dist = res["NSE"]
        SE_dist = res["SE"]
        sensitivity_dist = res["Sensitivity"]

        return {"Sensitivity": sensitivity_dist.detach().numpy(), "amp": amp, "NSE": NSE_dist.detach().numpy(), "SE": SE_dist.detach().numpy(), "period": period}

    # Test model performance on IQC RNNs
    name = 'iqc-rnn_w10_gamma0.0_n4'
    model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    res = test_training_dist(model)
    io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # name = 'iqc-rnn_w10_gamma3.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # res = test_training_dist(model)
    # io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # name = 'iqc-rnn_w10_gamma5.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # res = test_training_dist(model)
    # io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # name = 'iqc-rnn_w10_gamma8.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # res = test_training_dist(model)
    # io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # name = 'iqc-rnn_w10_gamma10.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # res = test_training_dist(model)
    # io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # name = 'iqc-rnn_w10_gamma15.0_n4'
    # model = iqcRNN.iqcRNN(nu, width, ny, width, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load(path + name + ".params"))
    # res = test_training_dist(model)
    # io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # lstm
    print("Running tests on LSTM")
    name = 'lstm_w10_gamma0.0_n4'
    model = lstm.lstm(nu, width, ny, layers=1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = test_training_dist(model)
    io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # rnn
    print("Running tests on RNN")
    name = 'rnn_w10_gamma0.0_n4'
    model = rnn.rnn(nu, width, ny, 1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = test_training_dist(model)
    io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # cirnn
    print("Running tests on cirnn")
    name = 'cirnn_w10_gamma0.0_n4'
    model = diRNN.diRNN(nu, width, ny, 1, nBatches=100)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = test_training_dist(model)
    io.savemat('./results/msd/training_stats/' + name + '.mat', res)

    # srnn
    print("Running tests on srnn")
    name = 'dnb_w10_gamma0.0_n4'
    model = dnb.dnbRNN(nu, width, ny, layers=1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = test_training_dist(model)
    io.savemat('./results/msd/training_stats/' + name + '.mat', res)
