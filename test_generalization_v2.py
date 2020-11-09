import torch
import numpy as np
# import data.load_data as load_data
from scipy.optimize import minimize, NonlinearConstraint
import multiprocessing
import scipy.io as io
import argparse
import os

import models.rnn as rnn
import models.lstm as lstm
import models.RobustRnn as RobustRnn
import models.ciRNN as ciRNN
import models.dnb as dnb

import data.msd_gen_data as msd

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4)
multiprocessing.set_start_method('spawn', True)


def test_performance(model, period, sd, samples=100, sim_len=1000, wash_per=200):

    # Normalization factors of the tan dataset.
    # mu_u = 0.42586177560540234
    # sigma_u = 0.17364211708359367
    # mu_y = 0.047316579744910395
    # sigma_y = 1.4127220752262701

    model.eval()
    sim = msd.msd_chain(N=2, spring_func=msd.tan_spring_func(), v_sd=0.0)
    batches = samples

    samples = []
    NSE = []
    SE = []
    for b in range(batches):
        u = msd.random_bit_stream(
            T=1000, period=100, u_sd=20.0)
        id_data = sim.simulate(u=u, Ts=1./10.)

        samples.append(id_data)

        with torch.no_grad():

            Yest = ((model(torch.Tensor(id_data["u"]).permute(
                0, 2, 1)).permute(0, 2, 1)))

            Y = torch.Tensor(id_data["y"])

            SE += [(Y[:, wash_per:, :] - Yest[:, wash_per:, :]
                    ).norm().detach()]
            NSE += [((Y[:, wash_per:, :] - Yest[:, wash_per:, :]).norm() /
                     Y[:, wash_per:, :].norm()).detach()]

            # plt.plot(Yest[0].detach())
            # plt.plot(Y[0].detach())
            # plt.show()

    res = {"NSE": NSE, "SE": SE}
    return res


if __name__ == "__main__":

    N = 4
    train_seq_len = 1000
    training_batches = 100
    mini_batch_size = 1

    nu = 1
    ny = 1
    width = 8
    neurons = 15
    batches = training_batches

    path = './results_v3/msd/'

    if not os.path.exists(path + 'lip/'):
        os.mkdir(path + 'lip/')
        os.mkdir(path + 'wcg/')
        os.mkdir(path + 'wcp/')
        print("Directory ", path, "lip/ Created")

    def vary_amplitude(model):
        print("\t Testing amplitude")
        samples = 20
        test_points = 21
        period = 100

        NSE_dist = np.zeros((test_points, samples))
        SE_dist = np.zeros((test_points, samples))
        amps = np.linspace(10.5, 40.5, test_points)

        for idx in range(test_points):
            res = test_performance(model, period, amps[idx], samples=samples)
            NSE_dist[idx, :] = res["NSE"]
            SE_dist[idx, :] = res["SE"]

        return {"amps": amps, "NSE": NSE_dist, "SE": SE_dist, "period": period}

    # # lstm
    print("Running tests on LSTM")
    name = 'lstm_w8_q5_gamma0.0'
    model = lstm.lstm(nu, width, ny, layers=1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    # rnn
    print("Running tests on RNN")
    name = 'rnn_w8_q0_gamma0.0'
    model = rnn.rnn(nu, width, ny, 1, nBatches=50)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    # Test generalization of Robust RNNs
    print("Running tests on robust-RNN")
    name = 'RobustRnn_w8_q15_gamma0.0'
    model = RobustRnn.RobustRnn(
        nu, width, ny, neurons, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results_v2/msd/generalization/amp_' + name + '.mat', res)

    name = 'RobustRnn_w8_q15_gamma0.5'
    model = RobustRnn.RobustRnn(
        nu, width, ny, neurons, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    name = 'RobustRnn_w8_q15_gamma1.0'
    model = RobustRnn.RobustRnn(
        nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    name = 'RobustRnn_w8_q15_gamma0.0'
    model = RobustRnn.RobustRnn(
        nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    name = 'iqc-rnn_w10_gamma10.0_n4'
    model = RobustRnn.RobustRnn(
        nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    name = 'iqc-rnn_w10_gamma15.0_n4'
    model = RobustRnn.RobustRnn(
        nu, width, ny, width, nBatches=batches, method='Neuron')
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    # rnn
    print("Running tests on RNN")
    name = 'rnn_w10_gamma0.0_n4'
    model = rnn.rnn(nu, width, ny, 1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    # cirnn
    print("Running tests on cirnn")
    name = 'cirnn_w10_gamma0.0_n4'
    model = ciRNN.ciRNN(nu, width, ny, 1, nBatches=100)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)

    # srnn
    print("Running tests on srnn")
    name = 'dnb_w10_gamma0.0_n4'
    model = dnb.dnbRNN(nu, width, ny, layers=1, nBatches=batches)
    model.load_state_dict(torch.load(path + name + ".params"))
    res = vary_amplitude(model)
    io.savemat('./results/msd/generalization/amp_' + name + '.mat', res)
