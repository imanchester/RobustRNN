
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


if __name__ == '__main__':



    loaders, lin_loader = f16.load_f16_data()
    stable_data = io.loadmat("./results/f16/RobustRnn_w75_gamma0.0_n4.mat")

    old_data = io.loadmat("./results/f16/RobustRnn_w75_gamma40.0_n4.mat")
    print("fin")