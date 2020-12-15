import torch
from torch import nn
from torch.nn import Parameter
from torch import Tensor

import cvxpy as cp
import numpy as np
import scipy as sp

import SIPPY.sippy as sippy
import torch_utils as utils
import matplotlib.pyplot as plt


class RobustRnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, res_size, method="Neuron",
                 nl=None, nBatches=1, learn_init_state=True, alpha=0, beta=1, supply_rate="stable"):
        super(RobustRnn, self).__init__()

        self.type = "iqcRNN"

        self.nx = hidden_size
        self.nu = input_size
        self.ny = output_size
        self.nw = res_size

        self.nBatches = nBatches
        # self.h0 = torch.nn.Parameter(torch.rand(nBatches, hidden_size))
        # self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size)) if learn_init_state else torch.zeros(nBatches, hidden_size)
        self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size))
        self.h0.requires_grad = learn_init_state

        self.criterion = torch.nn.MSELoss()

        #  nonlinearity
        self.nl = torch.nn.ReLU() if nl is None else nl

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

    def lipschitz_LMI(self, gamma=10.0, eps=1E-4):
        def l2gb_lmi():
            T = self.construct_T()
            M = utils.bmat([[-2 * self.alpha * self.beta * T, (self.alpha + self.beta) * T],
                            [(self.alpha + self.beta) * T, - 2 * T]])

            L_sq = gamma ** 2

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
            L_sq = gamma ** 2

            # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
            Mat11 = utils.bmat([[E + E.T - P, -self.beta * Ctild.T, zxu],
                                [-self.beta * Ctild, 2 * T, -self.beta * Dtild],
                                [zxu.T, -self.beta * Dtild.T, L_sq * torch.eye(self.nu)]])

            Mat21 = utils.bmat([[F, B1, B2], [C1, D11, D12]])
            Mat22 = utils.bmat([[P, torch.zeros((self.nx, self.ny))],
                                [torch.zeros((self.ny, self.nx)), torch.eye(self.ny)]])

            Mat = utils.bmat([[Mat11, Mat21.T],
                              [Mat21, Mat22]])

            return 0.5 * (Mat + Mat.T)

        def E_pd():
            return 0.5 * (self.E + self.E.T) - eps * torch.eye(self.nx)

        def P_pd():
            return 0.5 * (self.P + self.P.T) - eps * torch.eye(self.nx)

        return [l2gb_lmi, E_pd, P_pd]

    def initialize_lipschitz_LMI(self, gamma=10.0, eps=1E-4, init_var=1.5, solver="SCS"):
        solver_tol = 1E-4
        print("Initializing Lipschitz LMI ...")
        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nx)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nx), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        elif self.method == "Network":
            # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
            multis = cp.Variable((self.nx + 1) * self.nx //
                                 2, 'lambdas', nonneg=True)

            # return the (ii,jj)'th multiplier
            def get_multi(ii, jj): return multis[(ii * (ii + 1)) // 2 + jj]

            # Get the structured matrix in T
            Id = np.eye(self.nx)
            def e(ii): return Id[:, ii:ii + 1]
            def Tij(ii, jj): return e(
                ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj)
                    for ii in range(0, self.nx) for jj in range(0, ii + 1))
        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        # square of L2 gain
        # L_sq = cp.Variable((1, 1), "rho")
        L_sq = gamma ** 2

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        Bu = cp.Variable((self.nx, self.nu), 'Bu')
        C = cp.Variable((self.ny, self.nx), 'C')
        Dw = cp.Variable((self.ny, self.nx), 'Dw')
        Du = cp.Variable((self.ny, self.nu), 'Du')

        Bw = cp.Variable((self.nx, self.nw), 'Bw')

        Cv = np.random.normal(
            0, init_var / np.sqrt(self.nx), (self.nw, self.nx))

        Gamma_1 = sp.linalg.block_diag(Cv, np.eye(self.nw))
        Gamma_v = np.concatenate(
            [Gamma_1, np.zeros((2 * self.nw, self.nu))], axis=1)

        M = cp.bmat([[-2 * self.alpha * self.beta * T, (self.alpha +
                                                        self.beta) * T], [(self.alpha + self.beta) * T.T, - 2 * T]])

        # Construct final LMI.
        zxw = np.zeros((self.nx, self.nw))
        zxu = np.zeros((self.nx, self.nu))
        zwu = np.zeros((self.nw, self.nu))
        zww = np.zeros((self.nw, self.nw))

        # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
        Mat11 = cp.bmat([[E + E.T - P, zxw, zxu], [zxw.T, zww, zwu],
                         [zxu.T, zwu.T, L_sq * np.eye(self.nu)]]) - Gamma_v.T @ M @ Gamma_v

        Mat21 = cp.bmat([[F, Bw, Bu], [C, Dw, Du]])
        Mat22 = cp.bmat([[P, np.zeros((self.nx, self.ny))], [
                        np.zeros((self.ny, self.nx)), np.eye(self.ny)]])

        Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> solver_tol * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        # Just find a feasible point
        A = np.random.normal(
            0, init_var / np.sqrt(self.nx), (self.nx, self.nx))

        # Just find a feasible point
        objective = cp.Minimize(cp.norm(E @ A - Bw))

        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        elif solver == "SCS":
            prob.solve(solver=cp.SCS)
        else:
            print("Select valid sovler")

        print("Initilization Status: ", prob.status)

        # self.L_squared = torch.nn.Parameter(torch.Tensor(L_sq.value))
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.Bw.weight = Parameter(Tensor(Bw.value))
        self.Bu.weight = Parameter(Tensor(Bu.value))
        self.C.weight = Parameter(Tensor(C.value))
        self.Dw.weight = Parameter(Tensor(Dw.value))
        self.Du.weight = Parameter(Tensor(Du.value))
        self.Cv.weight = Parameter(Tensor(Cv))

    def init_lipschitz_ss(self, loader, gamma=10.0, eps=1E-4, init_var=1.2, solver="SCS"):

        print("RUNNING N4SID for intialization of A, Bu, C, Du")

        data = [(u, y) for (idx, u, y) in loader]
        U = data[0][0][0].numpy()
        Y = data[0][1][0].numpy()
        sys_id = sippy.system_identification(
            Y, U, 'N4SID', SS_fixed_order=self.nx)

        Ass = sys_id.A
        Bss = sys_id.B
        Css = sys_id.C
        Dss = sys_id.D

        # Calculate the trajectory.
        Xtraj = np.zeros((self.nx, Y.shape[1]))
        for tt in range(1, Y.shape[1]):
            Xtraj[:, tt:tt+1] = Ass @ Xtraj[:,
                                            tt - 1:tt] + Bss @ U[:, tt - 1:tt]

        # Sample points, calulate next state
        samples = 5000
        xtild = 3 * np.random.randn(self.nx, samples)
        utild = 3 * np.random.randn(self.nu, samples)
        xtild_next = Ass @ xtild + Bss @ utild

        print("Initializing using LREE")

        solver_tol = 1E-3
        print("Initializing stable LMI ...")

        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1, 1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nw)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        elif self.method == "Network":
            print(
                'YOU ARE USING THE NETWORK IQC MULTIPLIER. THIS DOES NOT WORK. PLEASE CHANGE TO NEURON OR LAYER')
            # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
            multis = cp.Variable((self.nx + 1) * self.nx //
                                 2, 'lambdas', nonneg=True)

            # Used for mapping vector to tril matrix
            indices = list(range((self.nx + 1) * self.nx // 2))
            Tril_Indices = np.zeros((self.nx, self.nx), dtype=int)
            Tril_Indices[np.tril_indices(self.nx)] = indices

            # return the (ii,jj)'th multiplier
            def get_multi(ii, jj): return multis[Tril_Indices[ii, jj]]

            # Get the structured matrix in T
            Id = np.eye(self.nx)
            def e(ii): return Id[:, ii:ii + 1]
            def Tij(ii, jj): return e(
                ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj)
                    for ii in range(self.nx) for jj in range(ii + 1))
        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        B1 = cp.Variable((self.nx, self.nw), 'Bw')
        B2 = cp.Variable((self.nx, self.nu), 'Bu')

        # Output matrices
        C1 = cp.Variable((self.ny, self.nx), 'C1')
        D11 = cp.Variable((self.ny, self.nw), 'D11')
        D12 = cp.Variable((self.ny, self.nu), 'D12')

        # Randomly initialize C2
        C2 = np.random.normal(
            0, init_var / np.sqrt(self.nw), (self.nw, self.nx))
        D22 = np.random.normal(
            0, init_var / np.sqrt(self.nw), (self.nw, self.nu))

        Ctild = T @ C2
        Dtild = T @ D22

        # lmi for dl2 gain bound.
        zxu = np.zeros((self.nx, self.nu))
        L_sq = gamma ** 2

        # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
        Mat11 = cp.bmat([[E + E.T - P, -self.beta * Ctild.T, zxu],
                         [-self.beta * Ctild, 2 * T, -self.beta * Dtild],
                         [zxu.T, -self.beta * Dtild.T, L_sq * np.eye(self.nu)]])

        Mat21 = cp.bmat([[F, B1, B2], [C1, D11, D12]])
        Mat22 = cp.bmat([[P, np.zeros((self.nx, self.ny))],
                         [np.zeros((self.ny, self.nx)), np.eye(self.ny)]])

        Mat = cp.bmat([[Mat11, Mat21.T],
                       [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> solver_tol * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        # Find the closest l2 gain bounded model
        bv = self.bv.detach().numpy()[:, None]

        if type(self.nl) is torch.nn.ReLU:
            wt = np.maximum(C2 @ xtild + D22 @ utild + bv, 0)
            wtraj = np.maximum(C2 @ Xtraj + D22 @ U + bv, 0)
        else:
            wt = np.tanh(C2 @ xtild + D22 @ utild + bv)
            wtraj = np.tanh(C2 @ Xtraj + D22 @ U + bv, 0)

        zt = np.concatenate([xtild_next, xtild, wt, utild], 0)

        EFBB = cp.bmat([[E, -F, -B1, -B2]])

        # empirical covariance matrix PHI
        Phi = zt @ zt.T
        R = cp.Variable((2 * self.nx + self.nw + self.nu,
                         2 * self.nx + self.nw + self.nu))
        Q = cp.bmat([[R, EFBB.T], [EFBB, E + E.T - np.eye(self.nx)]])

        # Add additional term for output errors

        eta = Y - C1 @ Xtraj - D11 @ wtraj - D12 @ U

        objective = cp.Minimize(cp.trace(R@Phi) + cp.norm(eta, p="fro")**2)
        constraints.append(Q >> 0)

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
        self.B1.weight = Parameter(Tensor(B1.value))
        self.B2.weight = Parameter(Tensor(B2.value))

        # Output mappings
        self.C1.weight = Parameter(Tensor(C1.value))
        self.D12.weight = Parameter(Tensor(D12.value))
        self.D11.weight = Parameter(Tensor(D11.value))
        self.by = Parameter(Tensor([0.0]))

        # Store Ctild and Dtild, C2 and D22 are extracted from
        #  T^{-1} \tilde{C} and T^{-1} \tilde{Dtild}
        self.C2tild = Parameter(Tensor(Ctild.value))
        self.Dtild = Parameter(Tensor(Dtild.value))

        print("Init Complete")

    def stable_LMI(self, eps=1E-4):
        def stable_lmi():
            T = self.construct_T()
            M = utils.bmat([[-2 * self.alpha * self.beta * T, (self.alpha + self.beta) * T],
                            [(self.alpha + self.beta) * T, - 2 * T]])

            # Construct LMIs
            P = self.P
            E = self.E
            F = self.F.weight
            B1 = self.B1.weight
            # Cv = self.Cv.weight

            S = utils.bmat([[torch.zeros(self.nx, self.nx), self.C2tild.T],
                            [self.C2tild, -2 * T]])

            # Construct final LMI.
            Mat11 = utils.block_diag(
                [E + E.T - P, torch.zeros(self.nw, self.nw)]) - S
            Mat21 = utils.bmat([[F, B1]])
            Mat22 = P

            Mat = utils.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])
            return 0.5 * (Mat + Mat.T)

        def E_pd():
            return 0.5 * (self.E + self.E.T) - eps * torch.eye(self.nx)

        def P_pd():
            return 0.5 * (self.P + self.P.T) - eps * torch.eye(self.nx)

        return [stable_lmi, E_pd, P_pd]

    def initialize_stable_LMI(self, eps=1E-4, init_var=1.5, obj='B', solver="SCS"):

        solver_tol = 1E-4
        print("Initializing stable LMI ...")
        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1, 1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nw)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        elif self.method == "Network":
            print(
                'YOU ARE USING THE NETWORK IQC MULTIPLIER. THIS DOES NOT WORK. PLEASE CHANGE TO NEURON OR LAYER')
            # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
            multis = cp.Variable((self.nx + 1) * self.nx //
                                 2, 'lambdas', nonneg=True)

            # Used for mapping vector to tril matrix
            indices = list(range((self.nx + 1) * self.nx // 2))
            Tril_Indices = np.zeros((self.nx, self.nx), dtype=int)
            Tril_Indices[np.tril_indices(self.nx)] = indices

            # return the (ii,jj)'th multiplier
            def get_multi(ii, jj): return multis[Tril_Indices[ii, jj]]

            # Get the structured matrix in T
            Id = np.eye(self.nx)
            def e(ii): return Id[:, ii:ii + 1]
            def Tij(ii, jj): return e(
                ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj)
                    for ii in range(self.nx) for jj in range(ii + 1))
        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        Bw = cp.Variable((self.nx, self.nw), 'Bw')

        Cv = np.random.normal(
            0, init_var / np.sqrt(self.nx), (self.nw, self.nx))
        Gamma_v = sp.linalg.block_diag(Cv, np.eye(self.nw))
        M = cp.bmat([[-2 * self.alpha * self.beta * T, (self.alpha +
                                                        self.beta) * T], [(self.alpha + self.beta) * T, - 2 * T]])

        # Construct final LMI.
        z1 = np.zeros((self.nx, self.nw))
        z2 = np.zeros((self.nw, self.nw))

        Mat11 = cp.bmat([[E + E.T - P, z1], [z1.T, z2]]) - \
            Gamma_v.T @ M @ Gamma_v

        Mat21 = cp.bmat([[F, Bw]])
        Mat22 = P

        Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (eps + solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        A = np.random.normal(
            0, init_var / np.sqrt(self.nx), (self.nx, self.nw))
        # Ass = np.eye(self.nx)

        # ensure wide distribution of eigenvalues for Bw
        objective = cp.Minimize(cp.norm(E @ A - Bw))

        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.SCS)

        print("Initilization Status: ", prob.status)

        # self.L_squared = torch.nn.Parameter(torch.Tensor(L_sq.value))
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.Bw.weight = Parameter(Tensor(Bw.value))
        self.Bu.weight = Parameter(Tensor(0.1 * self.Bu.weight.data))
        self.C.weight = Parameter(Tensor(0.1 * self.C.weight.data))
        self.Du.weight = Parameter(Tensor(0.0 * self.Du.weight.data))

        self.Ctild = Parameter(Tensor(Cv))

        print("Init Complete")

    def init_stable_ss(self, loader, eps=1E-4, init_var=1.2, solver="SCS"):

        print("RUNNING N4SID for intialization of A, Bu, C, Du ... ")
        solver_tol = 1E-4

        data = [(u, y) for (idx, u, y) in loader]
        U = data[0][0][0].numpy()
        Y = data[0][1][0].numpy()
        sys_id = sippy.system_identification(
            Y, U, 'N4SID', SS_fixed_order=self.nx)

        Ass = sys_id.A
        Bss = sys_id.B
        Css = sys_id.C
        Dss = sys_id.D

        x = np.zeros((self.nx, Y.shape[1]))
        for t in range(1, Y.shape[1]):
            x[:, t:t+1] = Ass @ x[:, t-1:t] + Bss @ U[:, t-1:t]

        Yest = Css @ x + Dss @ U

        NSE = np.linalg.norm(Yest - Y) / np.linalg.norm(Y)
        print('\t...Complete  NSE = {:1.3f}'.format(NSE))

        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1, 1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nw)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        # B1 = cp.Variable((self.nx, self.nw), 'Bw')

        B1 = np.zeros((self.nx, self.nw))
        B2 = cp.Variable((self.nx, self.nu), 'Bu')

        # Randomly initialize C2
        C2 = np.random.normal(
            0, init_var / np.sqrt(self.nw), (self.nw, self.nx))
        D22 = np.random.normal(
            0, init_var / np.sqrt(self.nw), (self.nw, self.nu))

        Ctild = T @ C2
        Dtild = T @ D22

        # Stability LMI
        S = cp.bmat([[np.zeros((self.nx, self.nx)), Ctild.T], [Ctild, -2 * T]])
        z1 = np.zeros((self.nx, self.nw))
        z2 = np.zeros((self.nw, self.nw))
        Mat11 = cp.bmat([[E + E.T - P, z1], [z1.T, z2]]) - S
        Mat21 = cp.bmat([[F, B1]])
        Mat22 = P
        Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (eps + solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-1]

        bv = self.bv.detach().numpy()[:, None]

        objective = cp.Minimize(cp.norm(E @ Ass - F))

        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.SCS)
        print("Initilization Status: ", prob.status)

        # Solve for output mapping from (W, X, U) -> Y
        # using linear least squares

        # Assign results to model
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.B1.weight = Parameter(Tensor(B1))
        self.B2.weight = Parameter(Tensor(E.value @ Bss))

        # Output mappings
        self.C1.weight = Parameter(Tensor(Css))
        self.D11.weight = Parameter(torch.zeros((self.ny, self.nw)))
        self.D12.weight = Parameter(Tensor(Dss))
        self.by = Parameter(Tensor([0.0]))

        # Store Ctild, C2 is extracted from T^{-1} \tilde{C}
        self.C2tild = Parameter(Tensor(Ctild.value))
        self.Dtild = Parameter(Tensor(Dtild.value))

        print("Init Complete")

    # def init_stable_ss(self, loader, eps=1E-4, init_var=1.2, solver="SCS"):

    #     print("RUNNING N4SID for intialization of A, Bu, C, Du")

    #     data = [(u, y) for (idx, u, y) in loader]
    #     U = data[0][0][0].numpy()
    #     Y = data[0][1][0].numpy()
    #     sys_id = sippy.system_identification(
    #         Y, U, 'N4SID', SS_fixed_order=self.nx)

    #     Ass = sys_id.A
    #     Bss = sys_id.B
    #     Css = sys_id.C
    #     Dss = sys_id.D

    #     x = np.zeros((self.nx, Y.shape[1]))
    #     for t in range(1, x.shape[1]):
    #         x[:, t:t+1] = Ass @ x[:, t-1:t] + Bss @ U[:, t-1:t]
    #     Yest = Css @ x + Dss @ U

    #     # Sample points, calulate next state
    #     samples = 5000
    #     xtild = 3 * np.random.randn(self.nx, samples)
    #     utild = 3 * np.random.randn(self.nu, samples)
    #     xtild_next = Ass @ xtild + Bss @ utild

    #     print("Initializing using LREE")

    #     solver_tol = 1E-4
    #     print("Initializing stable LMI ...")

    #     # Lip SDP multiplier
    #     if self.method == "Layer":
    #         multis = cp.Variable((1, 1), 'lambdas', nonneg=True)
    #         T = multis * np.eye(self.nw)

    #     elif self.method == "Neuron":
    #         multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
    #         T = cp.diag(multis)

    #     elif self.method == "Network":
    #         print(
    #             'YOU ARE USING THE NETWORK IQC MULTIPLIER. THIS DOES NOT WORK. PLEASE CHANGE TO NEURON OR LAYER')
    #         # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
    #         multis = cp.Variable((self.nx + 1) * self.nx //
    #                              2, 'lambdas', nonneg=True)

    #         # Used for mapping vector to tril matrix
    #         indices = list(range((self.nx + 1) * self.nx // 2))
    #         Tril_Indices = np.zeros((self.nx, self.nx), dtype=int)
    #         Tril_Indices[np.tril_indices(self.nx)] = indices

    #         # return the (ii,jj)'th multiplier
    #         def get_multi(ii, jj): return multis[Tril_Indices[ii, jj]]

    #         # Get the structured matrix in T
    #         Id = np.eye(self.nx)
    #         def e(ii): return Id[:, ii:ii + 1]
    #         def Tij(ii, jj): return e(
    #             ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

    #         # Construct the full conic comibation of IQC's
    #         T = sum(Tij(ii, jj) * get_multi(ii, jj)
    #                 for ii in range(self.nx) for jj in range(ii + 1))
    #     else:
    #         print("Invalid method selected. Try Neuron, Layer or Network")

    #     # Construct LMIs
    #     P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
    #     E = cp.Variable((self.nx, self.nx), 'E')
    #     F = cp.Variable((self.nx, self.nx), 'F')
    #     B1 = cp.Variable((self.nx, self.nw), 'Bw')
    #     B2 = cp.Variable((self.nx, self.nu), 'Bu')

    #     # Randomly initialize C2
    #     C2 = np.random.normal(
    #         0, init_var / np.sqrt(self.nw), (self.nw, self.nx))
    #     D22 = np.random.normal(
    #         0, init_var / np.sqrt(self.nw), (self.nw, self.nu))

    #     Ctild = T @ C2
    #     Dtild = T @ D22

    #     # Stability LMI
    #     S = cp.bmat([[np.zeros((self.nx, self.nx)), Ctild.T], [Ctild, -2 * T]])
    #     z1 = np.zeros((self.nx, self.nw))
    #     z2 = np.zeros((self.nw, self.nw))
    #     Mat11 = cp.bmat([[E + E.T - P, z1], [z1.T, z2]]) - S
    #     Mat21 = cp.bmat([[F, B1]])
    #     Mat22 = P
    #     Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

    #     # epsilon ensures strict feasability
    #     constraints = [Mat >> (eps + solver_tol) * np.eye(Mat.shape[0]),
    #                    P >> (eps + solver_tol) * np.eye(self.nx),
    #                    E + E.T >> (eps + solver_tol) * np.eye(self.nx),
    #                    multis >= 1E-6]

    #     # ensure wide distribution of eigenvalues for Bw
    #     bv = self.bv.detach().numpy()[:, None]

    #     if type(self.nl) is torch.nn.ReLU:
    #         wt = np.maximum(C2 @ xtild + D22 @ utild + bv, 0)
    #     else:
    #         wt = np.tanh(C2 @ xtild + D22 @ utild + bv)

    #     zt = np.concatenate([xtild_next, xtild, wt, utild], 0)

    #     EFBB = cp.bmat([[E, -F, -B1, -B2]])

    #     # empirical covariance matrix PHI
    #     Phi = zt @ zt.T
    #     R = cp.Variable((2*self.nx + self.nw + self.nu,
    #                      2*self.nx + self.nw + self.nu))
    #     Q = cp.bmat([[R, EFBB.T], [EFBB, E + E.T - np.eye(self.nx)]])

    #     objective = cp.Minimize(cp.trace(R@Phi))
    #     constraints.append(Q >> 0)

    #     prob = cp.Problem(objective, constraints)

    #     if solver == "mosek":
    #         prob.solve(solver=cp.MOSEK)
    #     else:
    #         prob.solve(solver=cp.SCS)
    #     print("Initilization Status: ", prob.status)

    #     # Solve for output mapping from (W, X, U) -> Y
    #     # using linear least squares
    #     X = np.zeros((self.nx, U.shape[1]))
    #     X[:, 0:1] = sys_id.x0

    #     Einv = np.linalg.inv(E.value)
    #     Ahat = Einv @ F.value
    #     Bwhat = Einv @ B1.value
    #     Buhat = Einv @ B2.value
    #     for t in range(1, U.shape[1]):
    #         w = np.maximum(C2 @ X[:, t-1:t] + D22 @ U[:, t-1:t] + bv, 0)
    #         X[:, t:t+1] = Ahat @ X[:, t-1:t] + Bwhat @ w + Buhat @ U[:, t-1:t]

    #     if type(self.nl) is torch.nn.ReLU:
    #         W = np.maximum(C2 @ X + D22 @ U + bv, 0)
    #     else:
    #         W = np.tanh(C2 @ X + D22 @ U + bv, 0)

    #     Z = np.concatenate([X, W, U], 0)
    #     output_mats = Y @ np.linalg.pinv(Z)

    #     C1 = output_mats[:, :self.nx]
    #     D11 = output_mats[:, self.nx:self.nx+self.nw]
    #     D12 = output_mats[:, self.nx+self.nw:]

    #     # Assign results to model
    #     self.IQC_multipliers = Parameter(Tensor(multis.value))
    #     self.E = Parameter(Tensor(E.value))
    #     self.P = Parameter(Tensor(P.value))
    #     self.F.weight = Parameter(Tensor(F.value))
    #     self.B1.weight = Parameter(Tensor(B1.value))
    #     self.B2.weight = Parameter(Tensor(B2.value))

    #     # Output mappings
    #     self.C1.weight = Parameter(Tensor(C1))
    #     self.D12.weight = Parameter(Tensor(D12))
    #     self.D11.weight = Parameter(Tensor(D11))
    #     self.by = Parameter(Tensor([0.0]))

    #     # Store Ctild, C2 is extracted from T^{-1} \tilde{C}
    #     self.C2tild = Parameter(Tensor(Ctild.value))
    #     self.Dtild = Parameter(Tensor(Dtild.value))

    #     print("Init Complete")

    def clone(self):
        copy = type(self)(self.nu, self.nx, self.ny, self.nw,
                          nBatches=self.nBatches, nl=self.nl, method=self.method)
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
