import torch
from torch import nn

from typing import List
from torch import Tensor

import cvxpy as cp
import numpy as np


class dnbRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1, nl=None, nBatches=1, init_var=1.0):
        super(dnbRNN, self).__init__()

        init_args = locals()

        self.nx = hidden_size
        self.nu = input_size
        self.ny = output_size
        self.layers = layers

        self.nBatches = nBatches
        self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size))

        #  nonlinearity
        if nl is None:
            self.nl = torch.nn.ReLU()
        else:
            self.nl = nl

        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.E0 = torch.eye(hidden_size).repeat((layers, 1, 1))

        # dynamic layers
        self.K = nn.ModuleList()
        self.H = nn.ModuleList()
        for layer in range(layers):
            self.K += [nn.Linear(input_size, hidden_size, bias=False)]
            self.H += [nn.Linear(hidden_size, hidden_size)]
            W = self.H[layer].weight.data.detach()

            [U, S, V] = W.svd()
            S[S >= 0.99] = 0.99
            self.H[layer].weight.data = U @ S.diag() @ V.T
            # self.H[layer].weight.data = torch.normal(0, init_var / (self.nx**0.5), (self.nx, self.nx))
            # self.K[layer].weight.data *= 0.1

    # @jit.script_method
    def forward(self, u, h0=None):

        inputs = u.permute(0, 2, 1)

        #  Initial state
        b = inputs.size(0)
        if h0 is None:
            ht = torch.zeros(b, self.nx)
        else:
            ht = h0

        Einv = torch.eye(self.nx).repeat((self.layers, 1, 1))
        # First calculate the inverse for E for each layer
        # Einv = []
        # for layer in range(self.layers):
        #     Einv += [self.E0[layer].inverse()]

        seq_len = inputs.size(1)
        outputs = torch.jit.annotate(List[Tensor], [ht])
        for tt in range(seq_len - 1):
            for layer in range(self.layers):
                xt = self.H[layer](ht) + self.K[layer](inputs[:, tt, :])
                eh = self.nl(xt)
                ht = eh.matmul(Einv[layer])

            outputs += [ht]

        states = torch.stack(outputs, 1)
        yest = self.output_layer(states)

        return yest.permute(0, 2, 1)

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

            ys = y - y.mean(1).unsqueeze(2)
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

    def project_norm_ball(self, eps=1E-3):

        for layer in range(self.layers):
            M = self.H[layer].weight.data
            U, S, V = torch.svd(M)

            S[S >= 1 - eps] = 1 - eps
            self.H[layer].weight.data = U @ S.diag() @ V.T

    # evaluates the contraction LMIs at the current parameter values - This is just an adapted from dirnn
    #  by setting the metric to Identity
    def norm_ball_lmi(self, eps):
        nx = self.nx

        lmis = []
        for layer in range(self.layers):

            def lmi(layer=layer):
                Hl = self.H[layer].weight
                El = torch.eye(self.nx)
                Pl = torch.eye(self.nx)
                Pn = torch.eye(self.nx)

                m1x = torch.cat([El + El.T - Pl, Hl.T], 1)
                m2x = torch.cat([Hl, Pn], 1)

                M = torch.cat([m1x, m2x], 0)
                return M - eps * torch.eye(2 * nx)
            lmis += [lmi]

        return lmis

    def clone(self):
        copy = type(self)(self.nu, self.nx, self.ny, layers=self.layers, nBatches=self.nBatches, nl=self.nl)
        copy.load_state_dict(self.state_dict())

        return copy

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

    def write_flat_params(self, x):
        index = 0
        theta = torch.Tensor(x)
        for p in self.parameters():
            p.data = theta[index:index + p.numel()].view_as(p.data)
            index = index + p.numel()

    def flatten_grad(self):
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