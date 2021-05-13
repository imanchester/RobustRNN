
import os
import argparse
import torch
import multiprocessing
import numpy as np
import scipy.io as io

# Import various models and data sets
import data.n_linked_msd as msd
import data.load_f16_data as f16
import opt.snlsdp_ipm as ipm
import opt.train as train
import models.ciRNN as ciRNN
import models.lstm as lstm
import models.rnn as rnn
import models.RobustRnn as RobustRnn
import models.dnb as dnb

import matplotlib.pyplot as plt


# Returns the performance metics for running model on loader
def test(model, loader):
    model.eval()

    # This is a pretty dodgy way of doing this.
    length = loader.__len__() * loader.batch_size
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

            error = yest.numpy() - y.numpy()
            
            N = error.shape[2]

            SE[idx] = (error ** 2 / N).sum(2) ** (0.5)
            # NSE[idx] = ((error ** 2).sum(1) / ) ** (0.5)
            NSE[idx] = np.linalg.norm(error, axis=(2)) / np.linalg.norm(y.numpy(), axis=(2))

    res = {"inputs": inputs, "outputs": outputs, "measured": measured, "SE": SE, "NSE": NSE}
    return res 


if __name__ == '__main__':

    nu = 2
    ny = 3
    width = 75
    neurons = 150
    batches = 504

    loaders, lin_loader = f16.load_f16_data()

    # # Test Robust RNNs
    # name = 'RobustRnn_w75_gamma40.0_n4'
    # model = RobustRnn.RobustRnn(nu, width, ny, neurons, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load("./results/f16/" + name + ".params"))
    # res = test(model, loaders["Test"])

    # # Test Robust RNNs
    # name = 'RobustRnn_w75_gamma40.0_n4'
    # model = RobustRnn.RobustRnn(nu, width, ny, neurons, nBatches=batches, method='Neuron')
    # model.load_state_dict(torch.load("./results/f16/" + name + ".params"))
    # test(model, name)


    # Load model
    stable_data = io.loadmat("./results/f16/RobustRnn_w75_gamma0.0_n4.mat")
    g40_data = io.loadmat("./results/f16/RobustRnn_w75_gamma40.0_n4.mat")
    g20_data = io.loadmat("./results/f16/RobustRnn_w75_gamma20.0_n4.mat")
    g10_data = io.loadmat("./results/f16/RobustRnn_w75_gamma10.0_n4.mat")

    print(stable_data["test"][0]["NSE"][0].mean())
    print(g40_data["test"][0]["NSE"][0].mean())
    print(g20_data["test"][0]["NSE"][0].mean())
    print(g10_data["test"][0]["NSE"][0].mean())
    print("fin")