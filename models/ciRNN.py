# deep implicit RNN.

import torch
from torch import nn
import matplotlib.pyplot as plt
# from torch.nn import Parameter
# import torch.jit as jit
# import models.dnb as dnb
from typing import List
from torch import Tensor

import cvxpy as cp
import numpy as np


class diRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1, nl=None, ar=False, nBatches=1, learn_init_state=True):
        super(diRNN, self).__init__()

        self.nx = hidden_size
        self.nu = input_size
        self.ny = output_size
        self.layers = layers

        self.nBatches = nBatches
        # self.h0 = torch.nn.Parameter(torch.rand(nBatches, hidden_size))
        # self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size)) if learn_init_state else torch.zeros(nBatches, hidden_size)
        self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size))
        self.h0.requires_grad = learn_init_state

        self.criterion = torch.nn.MSELoss()

        # autoregressive
        self.ar = ar

        #  nonlinearity
        if nl is None:
            self.nl = torch.nn.ReLU()
        else:
            self.nl = nl

        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        # self.D = torch.nn.Linear(input_size, output_size)

        # metric
        E0 = torch.eye(hidden_size).repeat((layers, 1, 1))
        P0 = torch.ones(layers, hidden_size)

        self.E = nn.Parameter(E0)
        self.P = nn.Parameter(P0)

        # Dynamics
        self.K = nn.ModuleList()
        self.H = nn.ModuleList()
        for layer in range(layers):
            self.K += [nn.Linear(input_size, hidden_size, bias=False)]
            self.K[layer].weight.data *= 0.1
            self.H += [nn.Linear(hidden_size, hidden_size)]
            self.H[layer].bias.data *= 0.1

    # @jit.script_method
    def forward(self, u, h0=None):

        inputs = u.permute(0, 2, 1)
        seq_len = inputs.size(1)

        #  Initial state
        b = inputs.size(0)
        if h0 is None:
            ht = torch.zeros(b, self.nx)
        else:
            ht = h0

        # First calculate the inverse for E for each layer
        Einv = []
        for layer in range(self.layers):
            Einv += [self.E[layer].inverse()]

        # Tensor to store the states in
        states = torch.zeros(b, self.layers * seq_len, self.nx)
        index = 1
        states[:, 0, :] = ht

        for tt in range(seq_len - 1):
            for layer in range(self.layers):
                # Update state
                xt = self.H[layer](ht) + self.K[layer](inputs[:, tt, :])
                eh = self.nl(xt)
                ht = eh @ Einv[(layer + 1) % self.layers]

                # Store state
                states[:, index, :] = ht
                index += 1

        # Take the outputs from the second last layer
        yest = self.output_layer(states[:, self.layers - 1::self.layers, :])

        return yest.permute(0, 2, 1)

    def check_storage(self, u1, u2, h0=None, gamma=0.5):

        inputs1 = u1.permute(0, 2, 1)
        inputs2 = u2.permute(0, 2, 1)

        #  Initial state
        b = inputs1.size(0)
        if h0 is None:
            ht1 = torch.zeros(b, self.nx)
            ht2 = torch.zeros(b, self.nx)

        # First calculate the inverse for E for each layer
        Einv = []
        for layer in range(self.layers):
            Einv += [self.E[layer].inverse()]

        seq_len = inputs1.size(1)
        states1 = torch.jit.annotate(List[Tensor], [ht1])
        states2 = torch.jit.annotate(List[Tensor], [ht2])

        # Vector to contain the storage function at each layer and timestep
        V = torch.zeros(seq_len * self.layers)
        sigma = torch.zeros(seq_len * self.layers)

        for tt in range(seq_len - 1):
            for layer in range(self.layers):

                # Calculate the storage function at this layer
                du = inputs1[:, tt, :] - inputs2[:, tt, :]
                dh = ht1 - ht2
                dy = self.output_layer(ht1) - self.output_layer(ht2)

                El = self.E[layer]
                Pl = torch.diag(self.P[layer])
                V[self.layers * tt + layer] = dh @ El @ Pl.inverse() @ El.T @ dh.T

                if layer == self.layers - 1:
                    sigma[self.layers * tt + layer] = du.norm() * gamma / self.layers
                else:
                    sigma[self.layers * tt + layer] = du.norm() * gamma / self.layers - dy.norm() / gamma

                # State update for first network
                xt1 = self.H[layer](ht1) + self.K[layer](inputs1[:, tt, :])
                eh1 = self.nl(xt1)
                ht1 = eh1 @ Einv[(layer + 1) % self.layers]

                # State update for Second network
                xt2 = self.H[layer](ht2) + self.K[layer](inputs2[:, tt, :])
                eh2 = self.nl(xt2)
                ht2 = eh2 @ Einv[(layer + 1) % self.layers]

                print("time{:d},layer{:d}, supply rate {:2.6f}, storage function {:2.6f}".format(tt, layer, sigma[self.layers * tt + layer], V[self.layers * tt + layer]))

            states1 += [ht1]
            states2 += [ht2]

        states1 = torch.stack(states1, 1)
        states2 = torch.stack(states2, 1)

        yest1 = self.output_layer(states1)
        yest2 = self.output_layer(states2)

        dy = yest1[0] - yest2[0]
        du = inputs1[0] - inputs2[0]

        du_mag = (du**2).sum(1)
        dy_mag = (du**2).sum(1)
        plt.plot(torch.cumsum(gamma ** 2 * du_mag - dy_mag, 0))

        return dy.permute(0, 2, 1)

    # Used for testing a model. Data should be a dictionary containing keys
    #   "inputs" and "outputs" in order (batches x seq_len x size)
    def test(self, data, h0=None):

        self.eval()
        with torch.no_grad():
            u = data["inputs"]
            y = data["outputs"]

            if h0 is None:
                h0 = self.h0

            yest, states = self.forward(u, h0=h0)

            ys = y - y.mean(1).unsqueeze(1)
            error = yest - y
            NSE = error.norm() / ys.norm()
            results = {"SE": float(self.criterion(y, yest)),
                       "NSE": float(NSE),
                       "estimated": yest.detach().numpy(),
                       "inputs": u.detach().numpy(),
                       "true_outputs": y.detach().numpy(),
                       "hidden_layers": self.nx,
                       "model": "lstm"}
        return results

    def init_l2(self, mu=0.05, epsilon=1E-2, init_var=1.5, custom_seed=None):
        n = self.nx
        constraints = []
        objective = 0.0

        if custom_seed is not None:
            np.random.seed(custom_seed)

        P = cp.Variable((self.layers, n), 'P')
        E = []
        H = []

        M = []
        for layer in range(self.layers):

            Id1 = np.eye(n)
            Id2 = np.eye(2 * n)

            # Initialize the forward dynamics
            W_star = np.random.normal(0, init_var / np.sqrt(n), (n, n))

            E += [cp.Variable((n, n), 'E{0:d}'.format(layer))]
            H += [cp.Variable((n, n), 'H{0:d}'.format(layer))]

            # Check to see if it is the last layer. If so, loop condition
            if layer == self.layers - 1:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[0])
            else:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[layer + 1])

            M += [cp.bmat([[E[layer] + E[layer].T - Pc - mu * Id1, H[layer].T],
                  [H[layer], Pn]])]

            constraints += [M[layer] >> epsilon * Id2]
            objective += cp.norm(W_star @ E[layer] - H[layer], "fro") ** 2

        prob = cp.Problem(cp.Minimize(objective), constraints)

        print("Beginning Initialization l2")
        prob.solve(verbose=False, solver=cp.MOSEK)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")

        print('Init Complete - status:', prob.status)

        # Reassign the values after projecting

        # Must convert from cvxpy -> numpy -> tensor
        E_np = np.stack(list(map(lambda M: M.value, E)), 0)
        P_np = np.stack(list(map(lambda M: M.value, P)), 0)

        self.E.data = torch.Tensor(E_np)
        self.P.data = torch.Tensor(P_np)

        for layer in range(self.layers):
            self.H[layer].weight.data = torch.Tensor(H[layer].value)

    def init_dl2(self, epsilon=1E-3, gamma=1e3, init_var=1.5, custom_seed=None):
        n = self.nx
        m = self.nu
        p = self.ny
        L = self.layers
        solver_res = 1E-5

        if custom_seed is not None:
            np.random.seed(custom_seed)

        constraints = []
        objective = 0.0

        P = cp.Variable((self.layers, n), 'P')
        E = []
        H = []
        B = []

        C = cp.Variable((p, n), 'C')

        M = []
        for layer in range(self.layers):

            # Initialize the forward dynamics
            W_star = np.random.normal(0, init_var / np.sqrt(n), (n, n))

            E += [cp.Variable((n, n), 'E{0:d}'.format(layer))]
            H += [cp.Variable((n, n), 'H{0:d}'.format(layer))]
            B += [cp.Variable((n, m), 'B{0:d}'.format(layer))]

            # Check to see if it is the last layer. If so, loop condition and include output
            if layer == self.layers - 1:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[0])
                M = cp.bmat([[E[layer] + E[layer].T - Pc, np.zeros((n, m)), H[layer].T, C.T],
                             [np.zeros((m, n)), gamma ** 2 / L * np.eye(m), B[layer].T, np.zeros((m, p))],
                             [H[layer], B[layer], Pn, np.zeros((n, p))],
                             [C, np.zeros((p, m)), np.zeros((p, n)), np.eye(p)]])
            else:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[layer + 1])
                M = cp.bmat([[E[layer] + E[layer].T - Pc, np.zeros((n, m)), H[layer].T],
                             [np.zeros((m, n)), gamma ** 2 / L * np.eye(m), B[layer].T],
                             [H[layer], B[layer], Pn]])

            q = M.shape[0]
            constraints += [M - (epsilon + solver_res) * np.eye(q) >> 0]
            objective += cp.norm(W_star @ E[layer] - H[layer], "fro") ** 2

        prob = cp.Problem(cp.Minimize(objective), constraints)

        print("Beginning Initialization l2")
        prob.solve(verbose=False, solver=cp.MOSEK, max_iters=5000)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")
        else:
            print('Init Complete - status:' + prob.status)

        # Reassign the values after projecting

        # Must convert from cvxpy -> numpy -> tensor
        E_np = np.stack(list(map(lambda M: M.value, E)), 0)
        P_np = np.stack(list(map(lambda M: M.value, P)), 0)
        self.output_layer.weight.data = torch.Tensor(C.value)

        self.E.data = torch.Tensor(E_np)
        self.P.data = torch.Tensor(P_np)

        for layer in range(self.layers):
            self.H[layer].weight.data = torch.Tensor(H[layer].value)
            self.K[layer].weight.data = torch.Tensor(B[layer].value)

    def project_l2(self, mu=0.05, epsilon=1E-5):

        n = self.nx

        constraints = []
        objective = 0.0

        P = cp.Variable((self.layers, n), 'P')
        E = []
        H = []

        M = []
        for layer in range(self.layers):

            Id1 = np.eye(n)
            Id2 = np.eye(2 * n)

            # Current values
            H_star = self.H[layer].weight.detach().numpy()

            W_star = H_star

            E += [cp.Variable((n, n), 'E{0:d}'.format(layer))]
            H += [cp.Variable((n, n), 'H{0:d}'.format(layer))]

            # Check to see if it is the last layer. If so, loop condition
            if layer == self.layers - 1:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[0])
            else:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[layer + 1])

            M += [cp.bmat([[E[layer] + E[layer].T - Pc - mu * Id1, H[layer].T],
                  [H[layer], Pn]])]

            constraints += [M[layer] >> epsilon * Id2]
            # objective += cp.norm(E[layer] - E_star, "fro") + cp.norm(H[layer] - H_star, "fro")
            objective += cp.norm(W_star @ E[layer] - H[layer], "fro") ** 2

        prob = cp.Problem(cp.Minimize(objective), constraints)
        # prob.solve(verbose=False, solver=cp.MOSEK)
        prob.solve(verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")

        print('projecting onto stable set')

        # Reassign the values after projecting

        # Must convert from cvxpy -> numpy -> tensor
        E_np = np.stack(list(map(lambda M: M.value, E)), 0)
        P_np = np.stack(list(map(lambda M: M.value, P)), 0)

        self.E.data = torch.Tensor(E_np)
        self.P.data = torch.Tensor(P_np)

        for layer in range(self.layers):
            self.H[layer].weight.data = torch.Tensor(H[layer].value)

    def contraction_lmi(self, mu=0.05, epsilon=1E-5):
        # evaluates the contraction LMIs at the current parameter values
        nx = self.nx

        lmis = []
        for layer in range(self.layers):

            def lmi(layer=layer):
                Hl = self.H[layer].weight
                El = self.E[layer]

                if layer == self.layers - 1:
                    Pl = torch.diag(self.P[layer])
                    Pn = torch.diag(self.P[0])
                else:
                    Pl = torch.diag(self.P[layer])
                    Pn = torch.diag(self.P[layer + 1])

                m1x = torch.cat([El + El.T - Pl - mu * torch.eye(nx), Hl.T], 1)
                m2x = torch.cat([Hl, Pn], 1)

                M = torch.cat([m1x, m2x], 0)
                return M - epsilon * torch.eye(2 * nx)
            lmis += [lmi]

            def E_pd_lmi(layer=layer):
                El = self.E[layer]
                return El + El.T - epsilon * torch.eye(El.size(0))
            lmis += [E_pd_lmi]

            def P_pd_lmi(layer=layer):
                Pl = torch.diag(self.P[layer])
                return Pl - epsilon * torch.eye(Pl.size(0))
            lmis += [P_pd_lmi]

        return lmis

    # Project onto differentially l2 gain bounded systems
    def project_dl2(self, gamma=1E2, epsilon=1E-2):

        #  Ensure that the inital point is on the interior. Greatly helps!
        solver_res = 1E-3

        n = self.nx
        m = self.nu
        p = self.ny
        L = self.layers

        constraints = []

        P = cp.Variable((self.layers, n), 'P')

        C_star = self.output_layer.weight.detach().numpy()
        C = cp.Variable((p, n), 'C')

        objective = cp.norm(C - C_star, "fro") ** 2

        # List of dec vars for each layer
        E = []
        H = []
        B = []

        for layer in range(self.layers):

            # Current values
            E_star = self.E[layer].detach().numpy()
            B_star = self.K[layer].weight.detach().numpy()
            H_star = self.H[layer].weight.detach().numpy()

            E += [cp.Variable((n, n), 'E{0:d}'.format(layer))]
            H += [cp.Variable((n, n), 'H{0:d}'.format(layer))]
            B += [cp.Variable((n, m), 'B{0:d}'.format(layer))]

            # Add terms for this layer to the objective
            objective += cp.norm(E[layer] - E_star, "fro")**2 + cp.norm(H[layer] - H_star, "fro")**2 + cp.norm(B[layer] - B_star, "fro") ** 2

            # Check to see if it is the last layer. If so, loop condition
            if layer == self.layers - 1:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[0])
                M = cp.bmat([[E[layer] + E[layer].T - Pc, np.zeros((n, m)), H[layer].T, C.T],
                             [np.zeros((m, n)), gamma / L * np.eye(m), B[layer].T, np.zeros((m, p))],
                             [H[layer], B[layer], Pn, np.zeros((n, p))],
                             [C, np.zeros((p, m)), np.zeros((p, n)), gamma * np.eye(p)]])
            else:
                Pc = cp.diag(P[layer])
                Pn = cp.diag(P[layer + 1])
                M = cp.bmat([[E[layer] + E[layer].T - Pc, np.zeros((n, m)), H[layer].T],
                             [np.zeros((m, n)), gamma / L * np.eye(m), B[layer].T],
                             [H[layer], B[layer], Pn]])

            constraints += [M >> (epsilon + solver_res) * np.eye(M.shape[0])]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        # prob.solve(verbose=False, solver=cp.MOSEK)
        prob.solve(verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")
        else:
            print('Projection onto feasible set succesful')

        E_np = np.stack(list(map(lambda M: M.value, E)), 0)
        P_np = np.stack(list(map(lambda M: M.value, P)), 0)

        self.E.data = torch.Tensor(E_np)
        self.P.data = torch.Tensor(P_np)
        self.output_layer.weight.data = torch.Tensor(C.value)

        # Reassign the values after projecting
        for layer in range(self.layers):
            self.H[layer].weight.data = torch.Tensor(H[layer].value)
            self.K[layer].weight.data = torch.Tensor(B[layer].value)

    def dl2_lmi(self, gamma=1e3, epsilon=1E-3):
        # evaluates the contraction LMIs at the current parameter values
        nx = self.nx
        m = self.nu
        p = self.ny
        L = self.layers

        lmis = []
        for layer in range(self.layers):

            def lmi(layer=layer):
                # Looks like there is a bug here due to the way layer is used
                Hl = self.H[layer].weight
                El = self.E[layer]
                Bl = self.K[layer].weight
                C = self.output_layer.weight

                if layer == self.layers - 1:
                    Pl = torch.diag(self.P[layer])
                    Pn = torch.diag(self.P[0])

                    m1x = torch.cat([El + El.T - Pl, torch.zeros(nx, m), Hl.T, C.T], 1)
                    m2x = torch.cat([torch.zeros(m, nx), gamma ** 2 / L * torch.eye(m), Bl.T, torch.zeros(m, p)], 1)
                    m3x = torch.cat([Hl, Bl, Pn, torch.zeros(nx, p)], 1)
                    m4x = torch.cat([C, torch.zeros(p, m), torch.zeros(p, nx), torch.eye(p)], 1)

                    M = torch.cat([m1x, m2x, m3x, m4x], 0)

                else:
                    Pl = torch.diag(self.P[layer])
                    Pn = torch.diag(self.P[layer + 1])

                    m1x = torch.cat([El + El.T - Pl, torch.zeros(nx, m), Hl.T], 1)
                    m2x = torch.cat([torch.zeros(m, nx), torch.tensor([[gamma ** 2 / L]]) * torch.eye(m), Bl.T], 1)
                    m3x = torch.cat([Hl, Bl, Pn], 1)

                    M = torch.cat([m1x, m2x, m3x], 0)

                return M - epsilon * torch.eye(M.size(1))
            lmis += [lmi]

            def E_pd_LMI(layer=layer):
                El = self.E[layer]
                return El + El.T - epsilon * torch.eye(El.size(1))
            lmis += [E_pd_LMI]

            def P_pd_LMI(layer=layer):
                Pl = torch.diag(self.P[layer])
                return Pl - epsilon * torch.eye(Pl.size(1))

            lmis += [P_pd_LMI]

        return lmis

    def clone(self):
        copy = type(self)(self.nu, self.nx, self.ny, nBatches=self.nBatches, layers=self.layers, nl=self.nl)
        copy.load_state_dict(self.state_dict())

        return copy

    # def make_explicit(self):
    #     new_model = dnb.dnbRNN(self.nu, self.nx, self.ny, self.layers, nBatches=self.nBatches)
    #     for layer in range(self.layers):
    #         new_model.K[layer].weight.data = self.K[layer].weight.data

    #         M = self.E[layer - 1].inverse()

    #         new_model.H[layer].weight.data = self.H[layer].weight.data @ M.T
    #         new_model.H[layer].bias.data = self.H[layer].bias.data

    #     # new_model.E0.data = self.E.data
    #     M = self.E[self.layers - 1].inverse()
    #     new_model.output_layer.weight.data = self.output_layer.weight.data @ M.T
    #     # new_model.output_layer.weight.data = self.output_layer.weight.data
    #     new_model.output_layer.bias.data = self.output_layer.bias.data

    #     return new_model

    def flatten_params(self):
        views = []
        for p in self.parameters():
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.reshape(-1)
            views.append(view)
        return torch.cat(views, 0)

    def write_flat_params(self, theta):
        index = 0

        for p in self.parameters():
            p.data = theta[index:index + p.numel()].view_as(p.data)
            index = index + p.numel()

    def double(self):
        for p in self.parameters():
            p.data = p.double()

        # assert index == self._numel()

    def check_dl2_LMIs(self, gamma, eps):
        LMIs = self.model.dl2_lmi(gamma=gamma, epsilon=eps)
        for (idx, lmi) in enumerate(LMIs):
            eigs = lmi().symeig()[0]
            if all(eigs >= 0.0):
                print("LMI ", idx, "checked: Okay!")
            else:
                print("LMI", idx, "not positive definite")

        for layer in range(self.layers):
            if all(self.model.E[layer].symeig()[0] > 1E-8):
                print("E[", layer, "] >0 checked: okay!")
            else:
                print("E[", layer, "] is n.d.")

            if all(self.model.P[layer] > 1E-8):
                print("P[", layer, "] >0 checked: okay!")
            else:
                print("P[", layer, "] is n.d.")
