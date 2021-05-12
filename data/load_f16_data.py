import torch
import csv as csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class IO_data(Dataset):

    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, U, X, Xnext):

        self.nu = U.shape[1]
        self.nx = X.shape[1]

        self.nBatches = U.shape[0]

        if torch.get_default_dtype() is torch.float32:
            convert = lambda x: x.astype(np.float32)
        else:
            convert = lambda x: x.astype(np.float64)

        # self.inputs = convert(np.concatenate([X, U], 1))
        self.U = U
        self.X = X
        # self.X0 = convert(X0)
        self.Xnext = convert(Xnext)

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        # return self.u[index][None, ...], self.y[index][None, ...]
        return index, [self.X[index], self.U[index]], self.Xnext[index]


class sim_IO_data(Dataset):
    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, U, X):

        self.nu = U.shape[1]
        self.nx = X.shape[1]

        self.nBatches = X.shape[0]

        if torch.get_default_dtype() is torch.float32:
            convert = lambda x: x.astype(np.float32)
        else:
            convert = lambda x: x.astype(np.float64)

        self.u = convert(U)
        self.X = convert(X)

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        return index, self.u[index], self.X[index]


def load_csv_data(fp):
    data = []
    with open(fp) as file:
        rows = csv.reader(file, delimiter=',', quotechar='"')
        next(rows, None) # skip header
        for row in rows:
            data.append([float(ri) for ri in row[0:5]])

    return data[:-1]



def load_f16_data(seq_len = 1024):
    fp1 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level1.csv"
    fp2 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level2_Validation.csv"
    fp3 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level3.csv"
    fp4 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level4_Validation.csv"
    fp5 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level5.csv"
    fp6 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level6_Validation.csv"
    fp7 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level7.csv"
    
    D4 = np.stack(load_csv_data(fp4))
    mu = np.mean(D4, axis=0)
    sigma = np.std(D4, axis=0)

    def standardize(xi): return (xi - mu[None, :]) / sigma[None,:]
    def split(D): return (D[:, 0:2], D[:, 2:])

    u1, y1 = split(standardize(np.stack(load_csv_data(fp1))))
    u2, y2 = split(standardize(np.stack(load_csv_data(fp2))))
    u3, y3 = split(standardize(np.stack(load_csv_data(fp3))))
    u4, y4 = split(standardize(np.stack(load_csv_data(fp4))))
    u5, y5 = split(standardize(np.stack(load_csv_data(fp5))))
    u6, y6 = split(standardize(np.stack(load_csv_data(fp6))))
    u7, y7 = split(standardize(np.stack(load_csv_data(fp7))))

    def format_inputs(seq_len, ui): return ui.reshape((-1,seq_len, 2)).transpose(0, 2, 1)
    def format_outputs(seq_len, yi): return yi.reshape((-1,seq_len, 3)).transpose(0, 2, 1)

    train_U = np.concatenate([format_inputs(1024, ui) for ui in [u1, u3, u5, u7]])
    train_Y = np.concatenate([format_outputs(1024, yi) for yi in [y1,y3, y5, y7]])

    val_U = np.concatenate([format_inputs(73728, ui) for ui in [u2,u4,u6]])
    val_Y = np.concatenate([format_outputs(73728, yi) for yi in [y2, y4, y6]])

    train_loader = DataLoader(sim_IO_data(train_U, train_Y), batch_size=4, shuffle=True, num_workers=4)
    # Only one batch
    val_loader = DataLoader(sim_IO_data(val_U, val_Y), batch_size=3, shuffle=False, num_workers=4)
    test_loader = DataLoader(sim_IO_data(val_U, val_Y), batch_size=1, shuffle=False, num_workers=4)
    linear_loader =  DataLoader(sim_IO_data(u1[None, :, :].transpose(0, 2, 1), y1[None, :, :].transpose(0, 2, 1)),
                                             batch_size=1, shuffle=True, num_workers=4)

    loaders = {"Training": train_loader, "Validation": val_loader, "Test": test_loader}
    return loaders, linear_loader