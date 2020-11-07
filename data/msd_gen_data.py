import torch
import os as os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interp
import scipy.io as io


class dataset(torch.utils.data.Dataset):
    def __init__(self, u, y):
        self.u = u
        self.y = y

    def __getitem__(self, idx):
        return (idx, self.u[idx], self.y[idx])

    def __len__(self):
        return self.u.shape[0]


def piecewise_spring_func(travel=2, kf=0.2):
    """
        Returns a spring func function that can can be called to generate the nonlinear spring profile.
        A piecewise linear spring function where the ratio of the central spring section to the
        outer spring section is kf.
    """
    def spring_func(x, travel=travel, kf=kf):
        d = travel / 2
        f = kf * x * (x < d) * (x > -d)
        f = f + (x - d + kf * d) * (x >= d)
        f = f + (x + d - kf * d) * (x <= -d)

        return f

    return spring_func


def tanh_spring_func():
    """
        Returns a tan spring func function that can can be called to generate the nonlinear spring profile.
    """
    def spring_func(x):
        return np.tanh(x)

    return spring_func


def tan_spring_func(travel=2.5):
    """
        Returns a tan spring func function that can can be called to generate the nonlinear spring profile.
    """
    def spring_func(x, travel=travel):
        return np.tan(np.pi * x / travel)

    return spring_func


def linear_spring_func():
    """
        Returns a tan spring func function that can can be called to generate the nonlinear spring profile.
    """
    def spring_func(x):
        return x

    return spring_func


def random_bit_stream(T, period, u_sd):
    """
    Generate a signal of length T with randomly varying period and randomly varying magnitude.
    """

    u_per = []
    while(sum(u_per) < T):
        u_per += [int((period * np.random.rand()))]

    # shorten the last element so that the periods add up to the total length
    u_per[-1] = u_per[-1] - (sum(u_per) - T)

    # Construct the actual signal
    u = np.concatenate(
        [u_sd * (np.random.rand() - 0.5) * np.ones((per, 1)) for per in u_per], 0)
    return u


class msd_chain():
    def __init__(self, N=5, spring_func=None, m=None, c=None, k=None):

        self.k = np.linspace(4.0, 2.5, N) if k is None else k
        self.c = np.linspace(1.0, 5.0, N) if c is None else c
        self.m = np.linspace(0.5, 1.0, N) if m is None else m

        self.v_sd = 0.1
        self.w_sd = 0.0

        self.N = N

        self.spring_func = spring_func if spring_func is not None else piecewise_spring_func()

    def dynamics(self, x, u, w=0):
        d = x[0::2, :]
        v = x[1::2, :]

        k = self.k
        c = self.c
        m = self.m

        # Vector containing the forces on each cart
        F = 0. * (d)

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

        dxdt = 0 * x
        dxdt[0::2] = v
        dxdt[1::2] = F / m[:, None]

        return dxdt

    def simulate(self, u, Ts):

        time = np.arange(0, u.shape[0]*Ts, Ts)
        Tend = time[-1]
        u_interp = interp.interp1d(time, u[:, 0])

        # Construct function for the dynamcis of the system
        x0 = np.zeros((2 * self.N))

        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            dx = dX.reshape(-1)
            return dx

        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time)
        t = sol['t']
        x = sol['y'].T

        y = sol['y'].T[:, -2:-1]

        y += self.v_sd * np.random.randn(*y.shape)

        id_data = {"time": t[None, ...], "u": u[None, ...],
                   "x": x[None, ...], "y": y[None, ...]}

        return id_data


def make_loader(file_path, train_batch_size):
    data = io.loadmat(file_path)

    def get_normalizer(data):
        mu_u = np.mean(data["u"][0, 0], axis=(0, 1))
        mu_y = np.mean(data["y"][0, 0], axis=(0, 1))

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

        return dataset(uhat, yhat)

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
            "Test": test_loader, "Linear_Init": init_data}


if __name__ == "__main__":

    N = 2
    batches = 50

    path = './data_v2/'
    print('Creating directory ...', path)
    try:
        os.mkdir(path)
    except OSError as err:
        print(err)

    print('generating data....')

    def gen_data(msd, name):
        print('\t\t\t ...', name)
        np.random.seed(1)

        train_samples = []
        for b in range(batches):
            u = random_bit_stream(T=1000, period=100, u_sd=20.0)
            id_data_train = msd.simulate(u=u, Ts=1./10.)
            train_samples.append(id_data_train)

        train_data = {}
        for key in id_data_train.keys():
            train_data[key] = np.concatenate(
                [x[key] for x in train_samples], 0)

        u = random_bit_stream(T=1000, period=100, u_sd=20.0)
        id_data_val = msd.simulate(u=u, Ts=1./10.)

        u = random_bit_stream(T=1000, period=100, u_sd=20.0)
        id_data_test = msd.simulate(u=u, Ts=1./10.)

        id_data = {"train": train_data,
                   "val": id_data_val, "test": id_data_test}

        io.savemat(name, id_data)

    # name = 'msd_tanh.mat'
    # msd = msd_chain(N=N, spring_func=tanh_spring_func())
    # gen_data(msd, path + name)

    # name = 'msd_linear.mat'
    # msd = msd_chain(N=N, spring_func=linear_spring_func())
    # gen_data(msd, path + name)

    name = 'msd_tan.mat'
    msd = msd_chain(N=N, spring_func=tan_spring_func())
    gen_data(msd, path + name)

    name = 'msd_piecewise.mat'
    msd = msd_chain(N=N, spring_func=piecewise_spring_func(travel=6, kf=0))
    gen_data(msd, path + name)

    print("~fin~")
