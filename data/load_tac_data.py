import torch
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset, DataLoader


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


def load(realization):

    data = io.loadmat('./data/Tac2017/pydata.mat')['processed_data']

    # Read data
    tu = data["train_u"][0, 0][realization][None, None, ...]
    ty = data["train_y"][0, 0][realization][None, None, ...]
    vu = data["val_u"][0, 0][realization][None, None, ...]
    vy = data["val_y"][0, 0][realization][None, None, ...]

    # make training data loader
    train_data = sim_IO_data(tu, ty)
    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=True, num_workers=1)

    # Make validation loader
    test_data = sim_IO_data(vu, vy)
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=True, num_workers=1)

    loaders = {"Training": train_loader,
               "Validation": train_loader,
               "Test": test_loader}

    return loaders
