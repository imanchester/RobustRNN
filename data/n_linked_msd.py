import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interp
from torch.utils.data import Dataset, DataLoader

import torch_utils as utils


class IO_data(Dataset):

    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, U, X, Xnext):

        self.nu = U.shape[1]
        self.nx = X.shape[1]

        self.nBatches = U.shape[0]

        if torch.get_default_dtype() is torch.float32:
            def convert(x): return x.astype(np.float32)
        else:
            def convert(x): return x.astype(np.float64)

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
            def convert(x): return x.astype(np.float32)
        else:
            def convert(x): return x.astype(np.float64)

        self.u = convert(U)
        self.X = convert(X)

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        return index, self.u[index], self.X[index]


class msd_chain():
    def __init__(self, N=5, T=100, Ts=0.1, u_sd=1, period=50, batchsize=1):
        # self.k = np.random.rand(N)
        # self.c = 1 * np.random.rand(N)
        # self.m = 1 + np.random.rand(N)

        self.k = np.linspace(1.0, 0.5, N)

        self.c = 0.5 * np.linspace(0.5, 1.0, N)
        self.m = 0.5 * np.linspace(0.5, 1, N)

        self.v_sd = 0.05
        self.w_sd = 0.0

        self.N = N
        self.u_sd = u_sd
        self.period = period
        self.travel = 2

        # number of samples
        self.T = T

        # Sampling period
        self.Ts = Ts

        self.batchsize = batchsize

    #  Generate a piecewise constant input with randomly varying period and magnitude
    def random_bit_stream(self, T=None):
        if T is None:
            T = self.T

        u_per = []
        while(sum(u_per) < T):
            u_per += [int((self.period * np.random.rand()))]

        # shorten the last element so that the periods add up to the total length
        u_per[-1] = u_per[-1] - (sum(u_per) - T)

        u = np.concatenate(
            [self.u_sd * (np.random.rand() - 0.5) * np.ones((per, 1)) for per in u_per], 0)
        return u

    # For displacements x, returns the spring force
    def spring_func(self, x):
        # epsilon = 1E-5
        # d = np.clip(x, -self.travel, self.travel)
        # return np.tan(np.pi * d / 2 / (self.travel + epsilon))

        # piecewise linear spring function
        kf = 0.25
        d = self.travel / 2
        f = kf * x * (x < d) * (x > -d)
        f = f + (x - d + kf * d) * (x >= d)
        f = f + (x + d - kf * d) * (x <= -d)

        return f

    # x is state size by batch size. u should be 1 by batch size
    def dynamics(self, x, u, w=0):
        d = x[0::2, :]
        v = x[1::2, :]

        if isinstance(x, torch.Tensor):
            k = torch.Tensor(self.k)
            c = torch.Tensor(self.c)
            m = torch.Tensor(self.m)
        else:
            k = self.k
            c = self.c
            m = self.m

        # Vector containing the forces on each cart
        # Automatically switches between tensors and np arrays
        F = 0 * (d)

        # Force on first cart first cart
        F[0] = F[0] + (w + u + k[0] * self.spring_func(-d[0])
                       + k[1] * self.spring_func(d[1] - d[0])
                       - c[0] * v[0] + c[1] * (v[1] - v[0]))

        # Force on the middle carts
        F[1:-1] = F[1:-1] + (k[1:-1, None] * self.spring_func(d[0:-2, :] - d[1:-1, :]) +
                             k[2:, None] * self.spring_func(d[2:, :] - d[1:-1, :]) +
                             c[1:-1, None] * (v[0:-2, :] - v[1:-1, :]) +
                             c[2:, None] * (v[2:, :] - v[1:-1, :]))

        # Force on the last cart
        F[-1] = F[-1] + (k[-1, None] * self.spring_func(d[-2, :] - d[-1, :])
                         + c[-1, None] * (v[-2, :] - v[-1, :]))

        # F[-1] = (self.k[-1, None] * self.spring_func(d[-2, :] - d[-1, :])
        #                  + self.c[-1, None] * (v[-2, :] - v[-1, :]))

        dxdt = 0 * x
        dxdt[0::2] = v
        dxdt[1::2] = F / m[:, None]

        return dxdt

    def simulate(self, u=None):

        # input and function to sample input
        if u is None:
            # End time of simulation
            Tend = self.T * self.Ts
            time = np.linspace(0, Tend, self.T)
            u = self.random_bit_stream()
        else:
            T = u.shape[0]
            Tend = self.Ts * T
            time = np.linspace(0, Tend, T)

        u_interp = interp.interp1d(time, u[:, 0])

        # Construct function for the dynamcis of the system
        # dyn = lambda t, x: self.dynamics(x, u_interp(t))
        x0 = np.zeros((2 * self.N))

        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time)
        t = sol['t']
        Y = sol['y'][None, -2:-1, :]

        # Add noise
        Y = Y + (np.random.normal(0, self.v_sd, Y.shape))
        u = u.T[:, None, :]

        # data = sim_IO_data(u, Y)
        # loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        # return loader

        return u, Y

    def torch_sim(self, u=None, x0=None):

        if x0 is None:
            x0 = torch.zeros(2 * self.N)

        states = torch.zeros(2 * self.N, u.shape[2])
        states[:, 0] = x0

        # Integrate using runge kutta scheme
        def dyn(x, u, t):
            return self.dynamics(x, u, w=0)

        for tt in range(1, u.shape[2]):
            states[:, tt:tt + 1] = utils.rk45(
                dyn, states[:, tt - 1:tt], u[0, 0, tt-1], tt * self.Ts, self.Ts)

        Y = states[None, -2:-1, :]
        Y = Y + 0*torch.Tensor(np.random.normal(0, self.v_sd, Y.shape))
        return Y

    # Grid the states space X x U with res points in all directions
    def grid_ss(self, res):
        # End time of simulation
        samples = 10
        Tend = samples * self.Ts
        time = np.linspace(0, samples * self.Ts, 10)

        # grid over control input
        u = [np.linspace(-1, 1, res)]
        # grid for each state
        x = [np.linspace(-1, 1, res) for n in range(2 * self.N)]

        X = np.meshgrid(*(x + u))  # unpack list into meshgrid

        U = X[-1].reshape(1, -1)
        X0 = np.stack([x.reshape(-1) for x in X[:-1]]).reshape(-1)

        # Construct function for the dynamcis of the system
        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, U)
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], X0, t_eval=time)
        t = sol['t']
        Y = sol['y'].reshape(2 * self.N, -1).T
        X = X0.reshape(2 * self.N, -1).T
        U = U.T

        data = IO_data(U, Y)

        # convert to data loader
        loader = DataLoader(data, batch_size=self.batchsize,
                            shuffle=True, num_workers=4)
        return loader

    def sim_ee(self, T=10000, mini_batch_size=100):

        # End time of simulation
        Tend = T * self.Ts
        time = np.linspace(0, Tend, T)

        # input and function to sample input
        u = self.random_bit_stream(T)
        u_interp = interp.interp1d(time, u[:, 0])

        # Construct function for the dynamcis of the system
        # dyn = lambda t, x: self.dynamics(x, u_interp(t))
        x0 = np.zeros((2 * self.N))

        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time)
        t = sol['t']
        x = sol['y']
        u = u.T

        U = u[:, :-1].T
        X = x[:, :-1].T
        Y = x[:, 1:].T

        # inputs = np.concatenate([X, U], 1)

        data = IO_data(U, X, Y)
        loader = DataLoader(data, batch_size=mini_batch_size,
                            shuffle=True, num_workers=4)
        return loader

    def short_sim(self, T=500, batches=100):
        # End time of simulation
        Tend = T * self.Ts
        time = np.linspace(0, Tend, T*Ts)

        # input and function to sample input
        u = self.random_bit_stream(T)
        u_interp = interp.interp1d(time, u[:, 0])

        # Construct function for the dynamcis of the system
        # dyn = lambda t, x: self.dynamics(x, u_interp(t))
        x0 = np.zeros((batches, 2 * self.N))

        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time)
        t = sol['t']
        x = sol['y']
        u = u.T

        U = u[:, :-1].T
        X = x[:, :-1].T
        Y = x[:, 1:].T + self.v_sd * (np.rand(x[:, 1:].T.shape) - 0.5)

        data = IO_data(U, Y)
        loader = DataLoader(data, batch_size=self.batchsize,
                            shuffle=False, num_workers=1)
        return loader

    # Grid the states space X x U with res points in all directions
    def sim_rand_ic(self, seq_len, batches, mini_batch_size=1):
        # End time of simulation
        samples = seq_len
        Tend = samples * self.Ts
        time = np.linspace(0, seq_len * self.Ts, seq_len)

        u = self.random_bit_stream(batches * seq_len)
        u = u.reshape(batches, seq_len)
        u_interp = interp.interp1d(time, u.T, axis=0)

        # grid for each state
        # unpack list into meshgrid
        X0 = 10 * (np.random.rand((2 * self.N * batches)) - 0.5)

        # Construct function for the dynamcis of the system
        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t))
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], X0, t_eval=time)
        t = sol['t']
        Y = sol['y'].reshape(2 * self.N, batches, -1).T
        X = X0.reshape(2 * self.N, -1).T
        U = u_interp(time)

        U = U.T[:, None, :]

        # extract position of last cart.
        Y = Y.transpose(1, 2, 0)[:, -2:-1, :]
        Y = Y + (np.random.normal(0, self.v_sd, Y.shape))

        data = sim_IO_data(U, Y)

        # convert to data loader
        loader = DataLoader(data, batch_size=mini_batch_size,
                            shuffle=True, num_workers=4)
        return loader


def load_saved_data(mini_batch_size=1):
    train_nl = np.load('./data/msd_dataset/train_u3.npy', allow_pickle=True)
    data = sim_IO_data(train_nl.item().get("u"), train_nl.item().get("y"))
    train_loader = DataLoader(
        data, batch_size=mini_batch_size, shuffle=True, num_workers=4)

    train_lin = np.load('./data/msd_dataset/train_u1.npy', allow_pickle=True)
    data = sim_IO_data(train_lin.item().get("u"), train_lin.item().get("y"))
    train_loader_lin = DataLoader(
        data, batch_size=mini_batch_size, shuffle=False, num_workers=4)

    val = np.load('./data/msd_dataset/val.npy', allow_pickle=True)
    data = sim_IO_data(val.item().get("u"), val.item().get("y"))
    val_loader = DataLoader(
        data, batch_size=mini_batch_size, shuffle=False, num_workers=4)

    # Load test data
    test_u1 = np.load('./data/msd_dataset/test_u1.npy', allow_pickle=True)
    test_u2 = np.load('./data/msd_dataset/test_u2.npy', allow_pickle=True)
    test_u3 = np.load('./data/msd_dataset/test_u3.npy', allow_pickle=True)
    test_u4 = np.load('./data/msd_dataset/test_u4.npy', allow_pickle=True)

    test_u = np.concatenate([test_u1.item()["u"], test_u2.item()["u"],
                             test_u3.item()["u"], test_u4.item()["u"]])

    test_y = np.concatenate([test_u1.item()["y"], test_u2.item()["y"],
                             test_u3.item()["y"], test_u4.item()["y"]])

    data = sim_IO_data(test_u, test_y)
    test_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

    loaders = {"Training": train_loader,
               "Validation": val_loader, "Test": test_loader}

    return loaders, train_loader_lin


def make_loader(file_path, train_batch_size):
    data = io.loadmat(file_path)

    m = data["train"]["u"][0, 0].shape[1]
    n = data["train"]["y"][0, 0].shape[1]

    def get_normalizer(data):
        mu_u = np.mean(data["u"][0, 0], axis=(0, 1))
        mu_y = np.mean(data["y"][0, 0], axis=(0, 1))
        # sigma_u = np.linalg.inv(np.linalg.cholesky(np.cov(data["u"][0, 0][0].T)))
        # sigma_y = np.linalg.inv(np.linalg.cholesky(np.cov(data["y"][0, 0][0].T)))

        sigma_u = 1. / np.std(data["u"][0, 0])
        sigma_y = 1. / np.std(data["y"][0, 0])

        return mu_u, sigma_u, mu_y, sigma_y

    mu_u, sigma_u, mu_y, sigma_y = get_normalizer(data["train"])

    # Reformat data into baches x seq_len x feature_size
    def make_set(dict, mu_u=None, sigma_u=None, mu_y=None, sigma_y=None):

        uhat = sigma_u * (dict["u"][0, 0].transpose(0,
                                                    2, 1)) - mu_u[None, None, ...]
        yhat = sigma_y * (dict["y"][0, 0].transpose(0,
                                                    2, 1)) - mu_y[None, None, ...]

        return msd.dataset(uhat, yhat)

    init_data = make_set(data["train"], sigma_u=sigma_u,
                         sigma_y=sigma_y, mu_u=mu_u, mu_y=mu_y)
    train_data = make_set(data["train"], sigma_u=sigma_u,
                          sigma_y=sigma_y, mu_u=mu_u, mu_y=mu_y)
    val_data = make_set(data["val"], sigma_u=sigma_u,
                        sigma_y=sigma_y, mu_u=mu_u, mu_y=mu_y)
    test_data = make_set(data["test"], sigma_u=sigma_u,
                         sigma_y=sigma_y, mu_u=mu_u, mu_y=mu_y)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(val_data, batch_size=1)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    return {"Training": train_loader, "Validation": val_loader,
            "Test": test_loader, "linear_init": init_data}
