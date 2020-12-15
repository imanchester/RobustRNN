import torch
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class sim_IO_data(Dataset):
    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, U, X):

        self.nu = U.shape[1]
        self.nx = X.shape[1]

        self.nBatches = X.shape[0]

        if torch.get_default_dtype() is torch.float32:
            def convert(x): return x.astype(np.float32)
        else:
            def convert(x): return x.astype(np.float64)

        self.u = convert(U)
        self.X = convert(X)

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        return index, self.u[index], self.X[index]


def load(train_realization, val_realization):

    data = io.loadmat('./data/Tac2017/pydata.mat')['data']

    # Unpack data from matlab data structure
    preamble = data[0][0][0]
    training = data[0][0][1]
    validation = data[0][0][2]

    # raw
    tu_raw = training[0, train_realization]['raw_data'][0, 0][0, 0]['u'][None,...]
    ty_raw = training[0, train_realization]['raw_data'][0, 0][0, 0]['y'][None,...]

    vu_raw = validation[0, val_realization]['raw_data'][0, 0][0, 0]['u'][None,...]
    vy_raw = validation[0, val_realization]['raw_data'][0, 0][0, 0]['y'][None,...]

    # filtered data
    tu_filt = training[0, train_realization]['filtered_data'][0, 0][0, 0]['u'][None,...]
    ty_filt = training[0, train_realization]['filtered_data'][0, 0][0, 0]['y'][None,...]
    tx_filt = training[0, train_realization]['filtered_data'][0, 0][0, 0]['x'][None,...]

    ssf = training[0, train_realization]['ssf']
    osf = training[0, train_realization]['osf']

    # make training data loader
    train_data = sim_IO_data(tu_filt, ty_filt)
    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=4)

    # Training raw
    train_data_raw = sim_IO_data(tu_raw, ty_raw)
    train_loader_raw = DataLoader(train_data_raw,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4)

    # Make validation from raw io data
    test_data = sim_IO_data(vu_raw, vy_raw)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)

    loaders = {"Training": train_loader,
               "Training_Raw": train_loader_raw,
               "Test": test_loader,
               "ssf": ssf,
               "osf": osf,
               "states": tx_filt}

    return loaders
