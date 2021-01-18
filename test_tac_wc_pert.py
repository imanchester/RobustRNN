import torch
import numpy as np
import multiprocessing
import scipy.io as io
import os

import models.rnn as rnn
import models.lstm as lstm
import models.RobustRnn as RobustRnn
import models.ciRNN as ciRNN
import models.dnb as dnb

import data.load_tac_data as data


import torch.nn.utils.clip_grad as clip_grad
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4)
multiprocessing.set_start_method('spawn', True)


# Numerically try and maximize the Lipschitz constant of the model. 
def test(model, test_loader, osf, alpha=1E-3, epsilon=0.1):
    tol_change = 1E-3

    model.eval()

    # Initialize signals
    u = torch.Tensor(test_loader.dataset.u)
    y = torch.Tensor(test_loader.dataset.X)

    # du = torch.randn((1, 1, 500), requires_grad=True)
    du = torch.randn_like(u, requires_grad=True)
    du.data = du.data * 0.05

    optimizer = torch.optim.Adam([u, du], lr=alpha)

    J_best = 0
    no_decrease_counter = 0

    for ii in range(5000):
        # Try and maximize lipschitz constant
        def closure():
            optimizer.zero_grad()
            yest = osf[0, 0][0, 0] * model(u + du)
            J = -((yest - y).norm()**2)
            J.backward()
            clip_grad.clip_grad_norm_([du], 100, 2)
            return J

        J = optimizer.step(closure)

        if du.norm() >= epsilon * u.norm():
            du.data = du.data * epsilon * u.norm() / du.norm()

        # Check for sufficient improvement in objective
        if J < J_best - tol_change:
            J_best = J.item()
            no_decrease_counter = 0

        else:
            no_decrease_counter += 1

        # If objective is not decreaseing decrease learning rate
        # or finish optimization if lr is too small
        if no_decrease_counter >= 40:
            no_decrease_counter = 0

            # If min lr is reached break
            lrs = [p['lr'] for p in optimizer.param_groups]
            if lrs[0] <= 1E-4:
                break

            print('reducing learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        grad_norm = du.grad.norm()
        print('iter', ii, '\tgamma = {:2.3f} \t g={:2.1f}'.format((-J).sqrt().item(), grad_norm.item()))

    yest_nom = osf[0, 0][0, 0] * model(u)
    yest_pert = osf[0, 0][0, 0] * model(u + du)

    NSE_nom = (yest_nom - y).norm() ** 2 / y.norm()**2
    NSE_pert = (yest_pert - y).norm() ** 2 / y.norm()**2

    res = {"u": u.detach().numpy(), "du": du.detach().numpy(),
           "yest_nom": yest_nom.detach().numpy(), "yest_pert": yest_pert.detach().numpy(),
           "y": y.detach().numpy(), "NSE_nom": NSE_nom, "NSE_pert": NSE_pert}

    # res = {"gamma": gamma, "u1": u.detach().numpy(), "u2": v.detach().numpy(),
    #        "y1": yest1.detach().numpy(), "y2": yest2.detach().numpy(), "True1": Y1, "True2": Y2,
    #        "NSE1": NSE1, "NSE2": NSE2}

    return res


def test_model(path, name, model,
               test_loader, osf, train_id, gamma):

    print("Testing and Saving Models")

    # Create target Directory if doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " Created")

    res_size = model.nw if hasattr(model, 'nw') else model.nx
    file_name = name + f'_w{model.nx}q{res_size}_gamma{gamma}_{train_id}.mat'

    # Test performance and store in dict for saving
    test_stats = test(model, test_loader, osf)

    data = {"test": test_stats,
            "nx": model.nx,
            "nu": model.nu,
            "ny": model.ny}

    # Save
    io.savemat(path + file_name, data)
    # torch.save(model.state_dict(), path + '/' + name + ".params")


def load_model(path, model, w, q, gamma, id):
    nu, ny = (1, 1)
    batches = 100

    # model should be a string containing lstm, rnn or RobustRnn
    # path = "./results/TAC_2017_test2/"
    name = model + f'_w{w}q{q}_gamma{gamma:1.1f}_{id}'

    if model == "RobustRnn":
        model = RobustRnn.RobustRnn(nu, w, ny, q, nBatches=batches, method='Neuron')
    elif model == "lstm":
        model = lstm.lstm(nu, w, ny, layers=1, nBatches=batches)
    else:
        model = rnn.rnn(nu, w, ny, 1, nBatches=batches)

    model.load_state_dict(torch.load(path + name + ".params"))

    return model


if __name__ == "__main__":

    # Don't care about the training loader. Need the validation loader.
    for train_id in range(0, 30):
        loaders = data.load(train_id, 0)

        model = load_model("./results/TAC_2017_test2/", "RobustRnn", 50, 50, 1.0, train_id)
        test_model("./results/TAC_2017_wcp/", "RobustRnn", model,
                   loaders["Test"], loaders["osf"], train_id, 1.0)

        model = load_model("./results/TAC_2017_test2/", "RobustRnn", 50, 50, 2.0, train_id)
        test_model("./results/TAC_2017_wcp/", "RobustRnn", model,
                   loaders["Test"], loaders["osf"], train_id, 2.0)

        model = load_model("./results/TAC_2017_test2/", "RobustRnn", 50, 50, 5.0, train_id)
        test_model("./results/TAC_2017_wcp/", "RobustRnn", model,
                   loaders["Test"], loaders["osf"], train_id, 5.0)

        # model = load_model("./results/TAC_2017_test2/", "RobustRnn", 50, 100, 0.0, train_id)
        # test_model("./results/TAC_2017_wcp/", "RobustRnn", model,
        #            loaders["Test"], loaders["osf"], train_id, 5.0)

        # Test Robust RNNs
        # model = load_model("./results/TAC_2017/", "rnn", 50, 100, 0.0, train_id)
        # test_model("./results/TAC_2017_wcp/", "rnn", model,
        #            loaders["Test"], loaders["osf"], train_id, 0.0)

        # model = load_model("./results/TAC_2017/", "lstm", 50, 100, 0.0, train_id)
        # test_model("./results/TAC_2017_wcp/", "lstm", model,
        #            loaders["Test"], loaders["osf"], train_id, 0.0)
