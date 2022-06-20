import numpy as np
import torch

from rbm.models import TMCRBM, TMCRBM2D

dtype = torch.float


def ImConcat(X, ncol=10, nrow=5, sx=28, sy=28, ch=1):
    """
    #TODO
    Tile a set of 2d arrays 
    """

    tile_X = []
    for c in range(nrow):
        L = torch.cat(
            (
                tuple(
                    X[i, :].reshape(sx, sy, ch)
                    for i in np.arange(c * ncol, (c + 1) * ncol)
                )
            )
        )
        tile_X.append(L)
    return torch.cat(tile_X, 1)


def ComputeProbabilityTMC1D(
    myRBM: TMCRBM,
    data: torch.Tensor,
    nb_chain: int,
    it_mcmc: int,
    it_mean: int,
    N: int,
    nb_point: int,
    border_length: float,
    V_g: torch.Tensor,
    β=1.0,
    device=torch.device("cpu"),
):
    start = torch.bernoulli(torch.rand(myRBM.Nv, nb_chain * nb_point, device=device))
    V_g = V_g[:, 0]
    if torch.mean(V_g) < 0:
        V_g = -V_g
    proj_data = torch.mv(data, V_g) / myRBM.Nv ** 0.5
    xmin = torch.min(proj_data) - border_length
    xmax = torch.max(proj_data) + border_length
    w_hat_b = torch.linspace(xmin, xmax, steps=nb_point, device=device)
    w_hat = torch.zeros(nb_chain * nb_point, device=device)
    for i in range(nb_point):
        for j in range(nb_chain):
            w_hat[i * nb_chain + j] = w_hat_b[i]
    _, _, vtab = myRBM.TMCSample(
        start, w_hat, N, V_g, it_mcmc=it_mcmc, it_mean=it_mean, β=β
    )
    myRBM.V0 = V_g
    myRBM.nb_chain = nb_chain
    myRBM.nb_point = nb_point
    myRBM.N = N
    myRBM.w_hat_b = w_hat_b
    res, p_m, grad_pot, w_hat_b = myRBM.compute_probability(vtab)
    return res, p_m, grad_pot, w_hat_b


def ComputeProbabilityTMC2D(
    myRBM: TMCRBM2D,
    data: torch.Tensor,
    nb_chain: int,
    it_mcmc: int,
    it_mean: int,
    N: int,
    nb_point_dim,
    border_length: float,
    V_g: torch.Tensor,
    device: torch.device,
    direction=torch.tensor([0, 1]),
    nDim=2,
    PCA=True,
    start=None,
):
    myRBM.device = device
    myRBM.border_length = border_length
    myRBM.N = N
    myRBM.nb_point_dim = nb_point_dim
    myRBM.nb_chain = nb_chain
    myRBM.gibbs_steps = it_mcmc
    myRBM.it_mean = it_mean
    myRBM.direction = direction
    myRBM.V0 = V_g
    myRBM.PCA = PCA

    if not PCA:
        _, _, V_g = torch.svd(myRBM.W)
        if torch.mean(V_g[:, 0]) < 0:
            V_g = -V_g
    proj_data = torch.mm(data.T, V_g).cpu() / myRBM.Nv ** 0.5
    limits = torch.zeros((2, nDim))
    for i in range(myRBM.direction.shape[0]):
        limits[0, i] = proj_data[:, myRBM.direction[i]].min() - border_length
        limits[1, i] = proj_data[:, myRBM.direction[i]].max() + border_length
    nb_point = nb_point_dim.prod()
    x_grid = np.linspace(limits[0, 0], limits[1, 0], nb_point_dim[0])
    x_grid = np.array([x_grid for i in range(nb_point_dim[1])])
    x_grid = x_grid.reshape(nb_point)
    y_grid = []
    y_d = np.linspace(limits[0, 1], limits[1, 1], nb_point_dim[1])
    for i in range(nb_point_dim[1]):
        for j in range(nb_point_dim[0]):
            y_grid.append(y_d[i])
    grid = torch.tensor([x_grid, y_grid], device=device)
    w_hat = torch.zeros((nDim, myRBM.nb_chain * myRBM.nb_point), device=myRBM.device)
    if start == None:
        start = torch.bernoulli(
            torch.rand(myRBM.Nv, nb_chain * nb_point, device=device)
        )

    myRBM.V0 = V_g
    myRBM.w_hat_b = grid
    myRBM.limits = limits
    w_hat = torch.zeros((2, nb_chain * nb_point), device=device)
    for i in range(nb_point):
        for j in range(nb_chain):
            w_hat[:, i * nb_chain + j] = myRBM.w_hat_b[:, i]
    # TMC Sampling
    _, _, vtab = myRBM.TMCSample(
        start,
        w_hat,
        myRBM.N,
        myRBM.V0,
        it_mcmc=myRBM.gibbs_steps,
        it_mean=myRBM.it_mean,
    )
    # Probability reconstruction
    square, p_m, w_hat_tmp, _, _ = myRBM.compute_probability(vtab)

    return square, p_m, w_hat_tmp

