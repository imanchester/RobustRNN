import torch


def block_diag(Mat):
    n = sum(M.shape[0] for M in Mat)
    m = sum(M.shape[1] for M in Mat)

    A = torch.zeros((n, m))

    index1 = 0
    index2 = 0
    for M in Mat:
        A[index1:index1 + M.shape[0], index2:index2 + M.shape[1]] = M
        index1 += M.shape[0]
        index2 += M.shape[1]
    return A


def bmat(Mat):
    Mix = [torch.cat(Mij, dim=1) for Mij in Mat]
    return torch.cat(Mix, dim=0)


# def rk45(f, xt, ut, t, h):

#     k1 = f(xt, ut, t)
#     k2 = f(xt + h / 2 * k1, ut, t + h / 2)
#     k3 = f(xt + h / 2 * k2, ut, t + h / 2)
#     k4 = f(xt + h * k3, ut, t + h)
#     xth = xt + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
#     return xth

def rk45(f, xt, ut, t, h):

    k1 = f(xt, ut, t)
    k2 = f(xt + h * 2 / 3 * k1, ut, t + h * 2 / 3)
    # k3 = f(xt + h / 2 * k2, ut, t + h / 2)
    # k4 = f(xt + h * k3, ut, t + h)
    xth = xt + h * (k1 + 3 * k2) / 4
    return xth

def rk45(f, xt, ut, t, h):
    k1 = f(xt, ut, t)
    k2 = f(xt + h * 2 / 3 * k1, ut, t + h * 2 / 3)
    # k3 = f(xt + h / 2 * k2, ut, t + h / 2)
    # k4 = f(xt + h * k3, ut, t + h)
    xth = xt + h * (k1 / 4 + 3 * k2 / 4)
    return xth
