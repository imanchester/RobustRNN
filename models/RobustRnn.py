import torch
from torch import nn
from torch.nn import Parameter
from torch import Tensor
import matplotlib.pyplot as plt
import SIPPY.sippy as sippy
import cvxpy as cp
import numpy as np
import scipy as sp

import torch_utils as utils


class RobustRnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, res_size, method="Neuron",
                 nl=None, alpha=0, beta=1, supply_rate="stable"):
        super(RobustRnn, self).__init__()

        self.type = "iqcRNN"

        self.nx = hidden_size
        self.nu = input_size
        self.ny = output_size
        self.nw = res_size

        # self.nBatches = nBatches

        self.criterion = torch.nn.MSELoss()

        #  nonlinearity
        # self.nl = torch.nn.ReLU() if nl is None else nl
        self.nl = torch.nn.ReLU()

        # metric
        E0 = torch.eye(hidden_size)
        P0 = torch.eye(hidden_size)

        # Metric
        self.E = nn.Parameter(E0)
        self.P = nn.Parameter(P0)

        # dynamics
        self.F = nn.Linear(hidden_size, hidden_size)
        self.F.bias.data = torch.zeros_like(self.F.bias.data)

        self.B1 = nn.Linear(res_size, hidden_size, bias=False)
        self.B2 = nn.Linear(input_size, hidden_size, bias=False)

        # output for v
        self.C2tild = Parameter(torch.randn(
            (self.nw, self.nx)) / np.sqrt(self.nx))
        self.bv = Parameter(torch.rand(self.nw) - 0.5)
        self.Dtild = Parameter(torch.randn(
            (self.nw, self.nu)) / np.sqrt(self.nu))

        # y ouputs for model
        self.C1 = nn.Linear(hidden_size, output_size, bias=False)
        self.D11 = nn.Linear(res_size, output_size, bias=False)
        self.D12 = nn.Linear(input_size, output_size, bias=False)
        self.by = Parameter(0*(torch.rand(self.ny) - 0.5))

        # Create parameters for the iqc multipliers
        self.alpha = alpha
        self.beta = beta
        self.method = method
        if method == "Layer":
            self.IQC_multipliers = torch.nn.Parameter(torch.zeros(1))

        elif method == "Neuron":
            self.IQC_multipliers = torch.nn.Parameter(1E-4*torch.ones(self.nw))

        elif method == "Network":
            # Same number of variables as in lower Triangular matrix?
            self.IQC_multipliers = torch.nn.Parameter(
                torch.zeros(((self.nw + 1) * self.nw) // 2))
        else:
            print("Do Nothing")

        self.supply_rate = supply_rate

    def forward(self, u, h0=None, c0=None):

        inputs = u.permute(0, 2, 1)
        seq_len = inputs.size(1)

        #  Initial state
        b = inputs.size(0)
        if h0 is None:
            ht = torch.zeros(b, self.nx)
        else:
            ht = h0

        # First calculate the inverse for E for each layer
        Einv = self.E.inverse()

        # Tensor to store the states in
        states = torch.zeros(b, seq_len, self.nx)
        yest = torch.zeros(b, seq_len, self.ny)
        index = 1
        states[:, 0, :] = ht

        # Construct C2 from Ctild.
        Cv = torch.diag(1 / self.IQC_multipliers) @ self.C2tild
        Dv = torch.diag(1 / self.IQC_multipliers) @ self.Dtild

        for tt in range(seq_len - 1):
            # Update state
            vt = ht @ Cv.T + inputs[:, tt, :] @ Dv.T + self.bv[None, :]

            wt = self.nl(vt)
            eh = self.F(ht) + self.B1(wt) + self.B2(inputs[:, tt, :])
            ht = eh @ Einv.T

            # Store state
            states[:, index, :] = ht
            index += 1

        # Output function
        W = self.nl(states @ Cv.T + inputs @ Dv.T + self.bv)
        yest = self.C1(states) + self.D11(W) + self.D12(inputs) + self.by

        return yest.permute(0, 2, 1)

    def construct_T(self):
        r'Returns a conic combination of IQC multipliers coupling different sets of neurons together.\
        Methods are listed in order of most scalable to most accurate.'
        # Lip SDP neuron
        if self.method == "Layer":
            # Tv = torch.cat([torch.ones(ni) * self.IQC_multipliers[idx] for (idx, ni) in enumerate(self.N[:-1])])
            T = torch.eye(self.nx) * self.IQC_multipliers

        elif self.method == "Neuron":
            T = torch.diag(self.IQC_multipliers)

        elif self.method == "Network":
            # return the (ii,jj)'th multiplier from the mulitplier vector
            def get_multi(
                ii, jj): return self.IQC_multipliers[ii * (ii + 1) // 2 + jj]

            # Get the structured matrix in T
            Id = torch.eye(self.nx)
            def e(ii): return Id[:, ii:ii + 1]
            def Tij(ii, jj): return e(
                ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj)
                    for ii in range(0, self.nx) for jj in range(0, ii + 1))

        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        return T

    #  Barrier function to ensure multipliers are positive
    def multiplier_barrier(self):
        return self.IQC_multipliers

    def stable_LMI(self, eps=1E-4):
        def stable_lmi():

            # Construct LMIs
            T = self.construct_T()

            P = self.P
            E = self.E
            F = self.F.weight
            B1 = self.B1.weight

            # v output
            Ctild = self.C2tild

            # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
            Mat11 = utils.bmat([[E + E.T - P, -self.beta * Ctild.T],
                                [-self.beta * Ctild, 2 * T]])

            Mat21 = utils.bmat([[F, B1]])
            Mat22 = utils.bmat([[P]])

            Mat = utils.bmat([[Mat11, Mat21.T],
                              [Mat21, Mat22]])

            return 0.5 * (Mat + Mat.T)

        def E_pd():
            return 0.5 * (self.E + self.E.T) - eps * torch.eye(self.nx)

        def P_pd():
            return 0.5 * (self.P + self.P.T) - eps * torch.eye(self.nx)

        return [stable_lmi, E_pd, P_pd]

    def init_stable(self, loader, eps=1E-4, solver="SCS"):

        print("RUNNING N4SID for intialization of A, Bu, C, Du")

        # data = [(u, y) for (idx, u, y) in loader]
        U = loader.u[0]
        Y = loader.y[0]
        sys_id = sippy.system_identification(
            Y, U, 'N4SID', SS_fixed_order=self.nx)

        Ass = sys_id.A
        Bss = sys_id.B
        Css = sys_id.C
        Dss = sys_id.D

        # Sample points, calulate next state
        x = np.zeros((self.nx, U.shape[1]))
        for t in range(1, U.shape[1]):
            x[:, t:t+1] = Ass @ x[:, t-1:t] + Bss @ U[:, t-1:t]
        plt.plot(Y[0])
        yest = Css @ x
        plt.plot(yest[0])
        plt.show()

        print("Initializing using LREE")

        solver_tol = 1E-3
        print("Initializing stable LMI ...")

        multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
        T = cp.diag(multis)

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        # B1 = cp.Variable((self.nx, self.nw), 'Bw')
        B1 = np.zeros((self.nx, self.nw))

        # Randomly initialize C2
        C2 = np.random.normal(0, 1. / np.sqrt(self.nw), (self.nw, self.nx))
        D22 = np.random.normal(0, 0. / np.sqrt(self.nw), (self.nw, self.nu))

        Ctild = T @ C2
        Dtild = T @ D22

        # Stability LMI
        Mat11 = cp.bmat([[E + E.T - P, -Ctild.T],
                         [-Ctild,       2*T]])

        Mat21 = cp.bmat([[F, B1]])
        Mat22 = P
        Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (eps + solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        # Projection objective
        objective = cp.Minimize(cp.norm(E @ Ass - F))
        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.SCS)

        print("Initilization Status: ", prob.status)

        # Assign results to model
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.B1.weight = Parameter(Tensor(B1))
        self.B2.weight = Parameter(self.E.detach() @ Bss)

        # Output mappings
        self.C1.weight = Parameter(Tensor(Css))
        self.D12.weight = Parameter(Tensor(Dss))
        self.D11.weight = Parameter(Tensor(torch.zeros(self.ny, self.nw)))
        self.by = Parameter(Tensor([0.0]))

        # Store Ctild, C2 is extracted from T^{-1} \tilde{C}
        self.C2tild = Parameter(Tensor(Ctild.value))
        self.Dtild = Parameter(Tensor(Dtild.value))

        print("Init Complete")

    def lipschitz_LMI(self, gamma=10.0, eps=1E-4):
        def l2gb_lmi():
            T = self.construct_T()

            # Construct LMIs
            P = self.P
            E = self.E
            F = self.F.weight
            B1 = self.B1.weight
            B2 = self.B2.weight

            # y output
            C1 = self.C1.weight
            D11 = self.D11.weight
            D12 = self.D12.weight

            # v output
            Ctild = self.C2tild
            Dtild = self.Dtild

            zxu = torch.zeros((self.nx, self.nu))

            # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
            Mat11 = utils.bmat([[E + E.T - P, -self.beta * Ctild.T, zxu],
                                [-self.beta * Ctild, 2 * T, -self.beta * Dtild],
                                [zxu.T, -self.beta * Dtild.T, gamma * torch.eye(self.nu)]])

            Mat21 = utils.bmat([[F, B1, B2], [C1, D11, D12]])
            Mat22 = utils.bmat([[P, torch.zeros((self.nx, self.ny))],
                                [torch.zeros((self.ny, self.nx)), gamma*torch.eye(self.ny)]])

            Mat = utils.bmat([[Mat11, Mat21.T],
                              [Mat21, Mat22]])

            return 0.5 * (Mat + Mat.T)

        def E_pd():
            return 0.5 * (self.E + self.E.T) - eps * torch.eye(self.nx)

        def P_pd():
            return 0.5 * (self.P + self.P.T) - eps * torch.eye(self.nx)

        return [l2gb_lmi, E_pd, P_pd]

    def init_l2(self, loader, gamma=1.0, eps=1E-4, solver="SCS"):

        print("RUNNING N4SID for intialization of A, Bu, C, Du")

        # data = [(u, y) for (idx, u, y) in loader]
        U = loader.u[0]
        Y = loader.y[0]
        sys_id = sippy.system_identification(
            Y, U, 'N4SID', SS_fixed_order=self.nx)

        Ass = sys_id.A
        Bss = sys_id.B
        Css = sys_id.C
        Dss = sys_id.D

        # Sample points, calulate next state
        x = np.zeros((self.nx, U.shape[1]))
        for t in range(1, U.shape[1]):
            x[:, t:t+1] = Ass @ x[:, t-1:t] + Bss @ U[:, t-1:t]
        plt.plot(Y[0])
        yest = Css @ x
        plt.plot(yest[0])
        plt.show()

        print("Initializing using LREE")

        solver_tol = 1E-3
        print("Initializing stable LMI ...")

        multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
        T = cp.diag(multis)

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        B1 = np.zeros((self.nx, self.nw))

        D11 = np.zeros((self.ny, self.nw))

        # Randomly initialize C2
        C2 = np.random.normal(0, 1. / np.sqrt(self.nw), (self.nw, self.nx))
        D22 = np.random.normal(0, 0. / np.sqrt(self.nw), (self.nw, self.nu))

        Ctild = T @ C2
        Dtild = T @ D22

        # Stability LMI
        zxu = np.zeros((self.nx, self.nu))

        Mat11 = cp.bmat([[E + E.T - P, -self.beta * Ctild.T, zxu],
                         [-self.beta * Ctild, 2 * T, -self.beta * Dtild],
                         [zxu.T, -self.beta * Dtild.T, gamma * np.eye(self.nu)]])

        Mat21 = cp.bmat([[F, B1, Bss], [Css, D11, Dss]])
        Mat22 = cp.bmat([[P, np.zeros((self.nx, self.ny))],
                         [np.zeros((self.ny, self.nx)), gamma * np.eye(self.ny)]])

        Mat = cp.bmat([[Mat11, Mat21.T],
                       [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (eps + solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        # Projection objective
        objective = cp.Minimize(cp.norm(E @ Ass - F))
        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.SCS)

        print("Initilization Status: ", prob.status)

        # Assign results to model
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.B1.weight = Parameter(Tensor(B1))
        self.B2.weight = Parameter(Tensor(Bss))

        # Output mappings
        self.C1.weight = Parameter(Tensor(Css))
        self.D12.weight = Parameter(Tensor(Dss))
        self.D11.weight = Parameter(Tensor(D11))
        self.by = Parameter(Tensor([0.0]))

        # Store Ctild, C2 is extracted from T^{-1} \tilde{C}
        self.C2tild = Parameter(Tensor(Ctild.value))
        self.Dtild = Parameter(Tensor(Dtild.value))

        print("Init Complete")

    def clone(self):
        copy = type(self)(self.nu, self.nx, self.ny, self.nw,
                          nl=self.nl, method=self.method)
        copy.load_state_dict(self.state_dict())

        return copy

    def flatten_params(self):
        r"""Return paramter vector as a vector x."""
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

    def write_flat_params(self, x):
        r""" Writes vector x to model parameters.."""
        index = 0
        theta = torch.Tensor(x)
        for p in self.parameters():
            p.data = theta[index:index + p.numel()].view_as(p.data)
            index = index + p.numel()

    def flatten_grad(self):
        r""" Returns vector of all gradients."""
        views = []
        for p in self.parameters():
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
