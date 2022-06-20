import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import torch

from rbm.models import TMCRBM, TMCRBM2D

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
dtype = torch.float


def loadTMCsave(path_file):
    """ Load the h5 archive at path_file and return it with the list of the saved times """
    if os.path.isfile(path_file):
        f_tmc = h5py.File(path_file, "r")
        alltimes = []
        for t in f_tmc["alltime"][:]:
            if "W" + str(t) in f_tmc:
                alltimes.append(t)
        alltimes = np.array(alltimes)
        return f_tmc, alltimes
    print(f"File not found at {path_file}")


def plot_valsing_evol(f, alltimes, nvalsing, scale="logx"):
    """ Plot the evolution of the singular values on a temporal logarithmic scale"""
    S = torch.zeros(nvalsing, len(alltimes), device=device)
    for i in range(len(alltimes)):
        t = alltimes[i]
        _, tmpS, tmpV = torch.svd(torch.tensor(f["W" + str(t)], device=device))
        if torch.mean(tmpV[:, 0]) < 0:
            tmpV = -tmpV
        S[:, i] = tmpS[:nvalsing]
    plt.plot(alltimes, S.T.cpu(), ".-")
    if scale == "logx":
        plt.semilogx()
    elif scale == "logy":
        plt.semilogy()
    elif scale == "loglog":
        plt.loglog()
    plt.title("Singular values of W during training")
    plt.xlabel("Time")
    plt.show()
    return S


def loadTMCRBM(
    f,
    time,
    lr=0.1,
    NGibbs=10,
    mb_s=500,
    num_pcd=500,
    PCA=True,
    direction=1,
    device=device,
    dtype=dtype,
):
    W = torch.tensor(f["W" + str(time)], device=device)
    Nh = W.shape[0]
    Nv = W.shape[1]
    RBM_TMC = TMCRBM(
        num_visible=Nv,
        num_hidden=Nh,
        device=device,
        lr=lr,
        gibbs_steps=NGibbs,
        UpdCentered=True,
        mb_s=mb_s,
        direction=direction,
        num_pcd=num_pcd,
        PCA=PCA,
    )
    RBM_TMC.W = W
    RBM_TMC.hbias = torch.tensor(f["hbias" + str(time)], device=RBM_TMC.device)
    RBM_TMC.vbias = torch.tensor(f["vbias" + str(time)], device=RBM_TMC.device)
    # RBM_TMC.X_pc = torch.tensor(f['X_pc'+str(time)+'0'], device = RBM_TMC.device)
    return RBM_TMC


def loadTMCRBM2D(
    f,
    time,
    lr=0.1,
    l2=0,
    NGibbs=10,
    annSteps=0,
    mb_s=500,
    num_pcd=500,
    ep_max=100,
    PCA=True,
    direction=torch.tensor([0, 1], device=device, dtype=dtype),
    nb_point_dim=torch.tensor([100, 100]),
    device=device,
    dtype=dtype,
):
    W = torch.tensor(f[f"W{time}"], device=device, dtype=dtype)
    Nh = W.shape[0]
    Nv = W.shape[1]
    RBM_TMC = TMCRBM2D(
        num_visible=Nv,
        num_hidden=Nh,
        device=device,
        dtype=dtype,
        lr=lr,
        gibbs_steps=NGibbs,
        UpdCentered=True,
        mb_s=mb_s,
        direction=direction,
        nb_point_dim=nb_point_dim,
        num_pcd=num_pcd,
        PCA=PCA,
    )
    RBM_TMC.W = W
    RBM_TMC.hbias = torch.tensor(
        f["hbias" + str(time)], device=RBM_TMC.device, dtype=RBM_TMC.dtype
    )
    RBM_TMC.vbias = torch.tensor(
        f["vbias" + str(time)], device=RBM_TMC.device, dtype=RBM_TMC.dtype
    )
    # RBM_TMC.X_pc = torch.tensor(f['X_pc'+str(time)+'0'], device = RBM_TMC.device)
    return RBM_TMC


def SampleTMC1D(p_m, w_hat_b, n_sample: int, region=None):
    """
    Sample the constraint from the reconstructed probability distribution
    
    p_m: the reconstructed distribution 
    
    w_hat_b: the discretization

    n_sample: the number of samples to be generated

    region: the region of the distribution to be sampled
        
    """
    if region == None:
        region = torch.zeros(2)
        region[0] = w_hat_b.min()
        region[1] = w_hat_b.max()
    cdf = np.zeros(len(p_m) - 1)
    for i in range(1, len(p_m)):
        cdf[i - 1] = simps(p_m[:i], w_hat_b[:i])

    good_sample = torch.zeros(n_sample)
    to_gen = n_sample
    while to_gen > 0:
        i = 0

        sample = torch.rand(to_gen)
        sample = sample.sort()[0]
        for k in range(len(cdf) - 1):
            while cdf[k + 1] > sample[i]:
                if w_hat_b[k] <= region[1] and w_hat_b[k] >= region[0]:
                    sample[i] = w_hat_b[k]
                else:
                    sample[i] = region[1] + 1
                i += 1
                if i == to_gen:
                    break

            if i == to_gen:
                break
        good = sample[sample != (region[1] + 1)]
        good_sample[
            (n_sample - to_gen) : min(
                n_sample - to_gen + good.shape[0], good_sample.shape[0]
            )
        ] = good[: min(good.shape[0], good_sample.shape[0])]
        to_gen -= good.shape[0]
    return good_sample


def genDataTMC1D(
    myRBM: TMCRBM,
    p_m: torch.Tensor,
    w_hat: torch.Tensor,
    n_sample: int,
    N: int,
    V,
    it_mcmc=30,
):
    """
    Generate data from the TMCRBM

    myRBM: TMCRBM 

    p_m: Tensor of shape ()
    the reconstructed distribution

    w_hat: Tensor of shape ()
        the discretization

    n_sample: int 
    the number of samples to be generated

    N: int
        The constraint on the gaussian bath

    V: Tensor of shape (Nv)
        The projection vector from the dataset space to the constrained dimension

    it_mcmc: int, default=30
        The number of iterations of the mcmc algorithm

    """
    gen_m = SampleTMC1D(p_m.cpu(), w_hat.cpu(), n_sample)
    vinit = torch.bernoulli(
        torch.rand((myRBM.Nv, n_sample), device=myRBM.device, dtype=myRBM.dtype)
    )
    tmpv, _, _ = myRBM.TMCSample(vinit, gen_m.cuda(), N, V, it_mcmc=it_mcmc)
    si, mi, _, _ = myRBM.Sampling(tmpv, it_mcmc=1)
    return si, mi


def SampleTMC2D(p_m, w_hat_b, n_sample, region=None):

    if region == None:
        region = torch.zeros(2, 2)
        region[0, 0] = w_hat_b[0].min()
        region[0, 1] = w_hat_b[0].max()
        region[1, 0] = w_hat_b[1].min()
        region[1, 1] = w_hat_b[1].max()

    p_y = np.zeros(p_m.shape[0])
    for i in range(1, len(p_y)):
        p_y[i - 1] = simps(p_m[:, i], w_hat_b[0, :, i])
    sample_y = SampleTMC1D(p_y, w_hat_b[1, 0, :], n_sample, region=region[1])
    sample_x = []
    for i in range(len(sample_y)):
        id_y = (torch.tensor(w_hat_b[1, 0, :]) >= sample_y[i]).nonzero(as_tuple=True)[
            0
        ][0]
        # print(id_y)
        sample_x.append(
            SampleTMC1D(
                p_m[:, id_y] / p_y[id_y - 1], w_hat_b[0, :, 1], 1, region=region[0]
            )[0]
        )
        # print(id_y,' ',sample_x[-1])
    return torch.stack(sample_x).reshape(len(sample_x)), sample_y


def genDataTMC2D(myRBM: TMCRBM2D, p_m, w_hat, n_sample, N, V, it_mcmc, region=None):
    """
    Generate data from the TMCRBM2D

    myRBM: TMCRBM 

    p_m: Tensor of shape ()
    the reconstructed distribution

    w_hat: Tensor of shape ()
        the discretization

    n_sample: int 
    the number of samples to be generated

    N: int
        The constraint on the gaussian bath

    V: Tensor of shape (Nv)
        The projection vector from the dataset space to the constrained dimension

    it_mcmc: int, default=30
        The number of iterations of the mcmc algorithm

    """
    x_grid, y_grid = SampleTMC2D(p_m, w_hat, n_sample, region=region)
    w_hat_b = torch.tensor(
        [x_grid.numpy(), y_grid.numpy()], device=myRBM.device, dtype=myRBM.dtype
    )
    vinit = torch.bernoulli(
        torch.rand((myRBM.Nv, n_sample), device=myRBM.device, dtype=myRBM.dtype)
    )
    n_chain = myRBM.nb_chain
    n_point = myRBM.nb_point
    myRBM.nb_chain = 1
    myRBM.nb_point = n_sample
    myRBM.direction = myRBM.direction.to(torch.long)
    tmpv, _, _ = myRBM.TMCSample(vinit, w_hat_b.cuda(), N, V, it_mcmc=it_mcmc)
    myRBM.nb_chain = n_chain
    myRBM.nb_point = n_point
    tmpv = tmpv.reshape(myRBM.Nv, n_sample)
    si, mi, _, _, = myRBM.Sampling(tmpv, it_mcmc=1)
    return si, mi
