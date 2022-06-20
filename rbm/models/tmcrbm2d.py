import torch
import numpy as np
import h5py
from scipy.integrate import simps

from rbm.definitions import MODEL_DIR
from rbm.models import RBM


class TMCRBM2D(RBM):
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        device: torch.device,
        gibbs_steps=10,
        var_init=1e-4,
        dtype=torch.float,
        num_pcd=100,
        lr=0.01,
        ep_max=100,
        mb_s=50,
        UpdCentered=True,
        CDLearning=False,
        ResetPermChainBatch=False,
        it_mean=8,
        nb_chain=15,
        N=20000,
        border_length=0.2,
        direction=torch.tensor([0, 1], dtype=torch.int),
        PCA=False,
        nb_point_dim=torch.tensor([100, 100]),
        save_fig=False,
    ):
        super().__init__(
            num_visible,
            num_hidden,
            device,
            gibbs_steps,
            var_init,
            dtype,
            num_pcd,
            lr,
            ep_max,
            mb_s,
            UpdCentered,
            CDLearning,
        )
        self.list_save_xpc = []
        self.ResetPermChainBatch = ResetPermChainBatch

        self.UpdFieldsVis = True
        self.UpdFieldsHid = True
        self.UpdWeights = True

        # TMC param
        self.nb_chain = nb_chain
        self.it_mean = it_mean
        self.N = N
        self.nb_point_dim = nb_point_dim.to(device)
        self.w_hat_tmp = np.zeros((2, self.nb_point_dim[0], self.nb_point_dim[1]))
        self.nb_point = self.nb_point_dim.prod()
        self.border_length = border_length

        self.save_fig = save_fig

        self.p_m = torch.zeros(self.nb_point - 1)
        self.w_hat_b = torch.zeros(self.nb_point)
        _, _, self.V0 = torch.svd(self.W)
        self.nDim = 2
        self.direction = direction
        self.PCA = PCA

        # permanent chain
        self.X_pc = torch.bernoulli(
            torch.rand(self.Nv, self.nb_chain * self.nb_point, device=self.device)
        )

    def TMCSample(
        self,
        v: torch.Tensor,
        w_hat: torch.Tensor,
        N: int,
        V: torch.Tensor,
        it_mcmc=0,
        it_mean=0,
        β=1,
    ):
        """
        Tethered Monte-Carlo Algorithm on the visible units

        Parameters:
        -----------
        v : torch.tensor of shape (Nv, nb_point*nb_chain)
            starting point for the Markov chains

        w_hat : torch.tensor of shape (nb_point*nb_chain)
            Constraint parameter values

        N : int #TODO
        
        V : torch.tensor of shape (Nv, :)
            The projection vector from the dataset space to the constrained dimension
        
        it_mcmc : int, default=0
            The number of MCMC steps.
            If set to 0, use the class variable self.it_mcmc instead.
        
        it_mean : int, default=0
            The number of MCMC steps used for the temporal mean.
            If set to 0, use the class variable self.it_mean instead.
        
        β : float, default=1
            Effective temperature. Used for annealing.
        
        Returns:
        --------
        v_curr : torch.tensor of shape (Nv, nb_point*nb_chain)
            The last visible value of the MC chains.

        h_curr : torch.tensor of shape (Nh, nb_point*nb_chain)
            The last hidden value of the MC chains.

        vtab : torch.tensor of shape (Nv, nb_point*nb_chain)
            The temporal mean over the last it_mean visible values for each MC chain.
        """
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps
        if it_mean == 0:
            it_mean = self.it_mean

        vtab = torch.zeros(v.shape, device=self.device)
        v_curr = v
        norm = 1 / (v_curr.shape[0] ** 0.5)
        w_curr = (torch.mm(v_curr.T, V) * norm)[
            :, self.direction
        ]  # Current tethered weight
        for t in range(it_mcmc):
            h_curr, _ = self.SampleHiddens01(v_curr)
            h_i = torch.mm(self.W.T, h_curr) + self.vbias.reshape(
                v.shape[0], 1
            )  #  Nv x Ns
            w_next = w_curr.clone()

            v_next = torch.clone(v_curr)
            for i in range(v_curr.shape[0]):
                v_next[i, :] = 1 - v_curr[i, :]  # Proposed change
                for j in range(w_next.shape[1]):
                    w_next[:, j] += (
                        (2 * v_next[i, :] - 1) * V[i, self.direction[j]] * norm
                    )  # New tethered weight

                # Compute -DeltaE
                ΔE = β * ((2 * v_next[i, :] - 1) * h_i[i, :]) - (N / 2) * (
                    torch.sum((w_hat.T - w_next) ** 2, dim=1)
                    - torch.sum((w_hat.T - w_curr) ** 2, dim=1)
                )

                tir = torch.rand(v_curr.shape[1], 1).to(self.device).squeeze()
                prob = torch.exp(ΔE).squeeze()
                # Update chains position with probability prob
                v_curr[i, :] = torch.where(tir < prob, v_next[i, :], v_curr[i, :])
                v_next[i, :] = torch.where(tir < prob, v_next[i, :], 1 - v_next[i, :])
                neg_index = torch.ones(w_curr.shape[0], dtype=bool)
                index = torch.where(tir < prob)[0]
                neg_index[index] = False
                w_curr[index, :] = w_next[index, :]
                w_next[neg_index, :] = w_curr[neg_index, :]
            # Temporal mean over the last it_mean iterations
            if t >= (it_mcmc - it_mean):
                vtab += v_curr
        vtab = vtab * (1 / it_mean)
        vtab = vtab.reshape(self.Nv, self.nb_point, self.nb_chain)
        v_curr = v_curr.reshape(self.Nv, self.nb_point, self.nb_chain)
        h_curr = h_curr.reshape(self.Nh, self.nb_point, self.nb_chain)
        return v_curr, h_curr, vtab

    def TMCSampleAnalysis(self, v, w_hat, N, V, it_mcmc=0, it_mean=0, β=1):
        """
        Tethered Monte-Carlo Algorithm on the visible units

        Parameters:
        -----------
        v : torch.tensor of shape (Nv, nb_point*nb_chain)
            starting point for the Markov chains

        w_hat : torch.tensor of shape (nb_point*nb_chain)
            Constraint parameter values

        N : #TODO
        
        
        V : torch.tensor of shape (Nv, :)
            The projection vector from the dataset space to the constrained dimension
        
        it_mcmc : int, default=0
            The number of MCMC steps.
            If set to 0, use the class variable self.it_mcmc instead.
        
        it_mean : int, default=0
            The number of MCMC steps used for the temporal mean.
            If set to 0, use the class variable self.it_mean instead.
        
        β : float, default=1
            effective temperature. Used for annealing
        Returns:
        --------

        """
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps
        if it_mean == 0:
            it_mean = self.it_mean

        vtab = torch.zeros(v.shape, device=self.device)
        v_curr = v
        norm = 1 / (v_curr.shape[0] ** 0.5)
        w_curr = (torch.mm(v_curr.T, V) * norm)[
            :, self.direction
        ]  # Current tethered weight
        save_proj = torch.zeros((it_mcmc, w_hat.shape[1], 2))
        print(w_curr.shape)
        for t in range(it_mcmc):
            h_curr, _ = self.SampleHiddens01(v_curr)
            h_i = torch.mm(self.W.T, h_curr) + self.vbias.reshape(
                v.shape[0], 1
            )  #  Nv x Ns
            w_next = w_curr.clone()

            v_next = torch.clone(v_curr)
            for i in range(v_curr.shape[0]):
                v_next[i, :] = 1 - v_curr[i, :]  # Proposed change
                for j in range(w_next.shape[1]):
                    w_next[:, j] += (
                        (2 * v_next[i, :] - 1) * V[i, self.direction[j]] * norm
                    )  # New tethered weight

                #  On calcul -DeltaE
                ΔE = β * ((2 * v_next[i, :] - 1) * h_i[i, :]) - (N / 2) * (
                    torch.sum((w_hat.T - w_next) ** 2, dim=1)
                    - torch.sum((w_hat.T - w_curr) ** 2, dim=1)
                )

                tir = torch.rand(v_curr.shape[1], 1).to(self.device).squeeze()
                prob = torch.exp(ΔE).squeeze()
                # Update chains position with probability prob
                v_curr[i, :] = torch.where(tir < prob, v_next[i, :], v_curr[i, :])
                v_next[i, :] = torch.where(tir < prob, v_next[i, :], 1 - v_next[i, :])
                neg_index = torch.ones(w_curr.shape[0], dtype=bool)
                index = torch.where(tir < prob)[0]
                neg_index[index] = False
                w_curr[index, :] = w_next[index, :]
                w_next[neg_index, :] = w_curr[neg_index, :]
            # Temporal mean over the last it_mean iterations
            if t >= (it_mcmc - it_mean):
                vtab += v_curr
            save_proj[t] = w_curr.clone()
        vtab = vtab * (1 / it_mean)
        vtab = vtab.reshape(self.Nv, self.nb_point, self.nb_chain)
        v_curr = v_curr.reshape(self.Nv, self.nb_point, self.nb_chain)
        h_curr = h_curr.reshape(self.Nh, self.nb_point, self.nb_chain)
        return v_curr, h_curr, vtab, save_proj

    def updateWeights(self, v_pos, h_pos, negTermV, negTermH, negTermW):
        """
        
        Inputs:
        -------

        """
        lr_p = self.lr / self.mb_s
        lr_n = self.lr
        self.W += h_pos.mm(v_pos.t()) * lr_p - negTermW * lr_n
        self.vbias += torch.sum(v_pos, 1) * lr_p - negTermV * lr_n
        self.hbias += torch.sum(h_pos, 1) * lr_p - negTermH * lr_n

    def updateWeightsCentered(self, v_pos, h_pos_v, h_pos_m, v_neg, h_neg_v, h_neg_m):
        """
        
        Inputs:
        -------
        """
        self.VisDataAv = torch.mean(v_pos, 1).float()
        self.HidDataAv = torch.mean(h_pos_m, 1).float()

        NormPos = 1.0 / self.mb_s

        Xc_pos = (v_pos.t() - self.VisDataAv).t()
        Hc_pos = (h_pos_m.t() - self.HidDataAv).t()

        si_neg = v_neg
        τa_neg = h_neg_v
        ΔW_neg = (
            h_neg_m
            - torch.outer(h_neg_v.float(), self.VisDataAv)
            - torch.outer(self.HidDataAv, v_neg)
            + torch.outer(self.HidDataAv, self.VisDataAv)
        )
        ΔW = Hc_pos.mm(Xc_pos.t()) * NormPos - ΔW_neg

        if self.UpdWeights:
            self.W += ΔW * self.lr

        ΔVB = (
            torch.sum(v_pos, 1) * NormPos
            - si_neg
            - torch.mv(ΔW.t().float(), self.HidDataAv)
        )
        if self.UpdFieldsVis:
            self.vbias += self.lr * ΔVB

        ΔHB = (
            torch.sum(h_pos_m, 1) * NormPos
            - τa_neg
            - torch.mv(ΔW.float(), self.VisDataAv)
        )
        if self.UpdFieldsHid:
            self.hbias += self.lr * ΔHB

    def compute_probability(self, vtab: torch.Tensor):
        """
        Compute the probability of the RBM

        Inputs:
        -------
        vtab: torch.tensor of shape (#TODO)
            The temporal mean of MC Chains of the TMC2D algorithm
        
        Outputs:
        --------
        square: torch.tensor of shape (#TODO) 

        p_m: torch.tensor of shape (#TODO)

        w_hat_tmp

        grad_pot: torch.tensor of shape (#TODO)

        w_hat_dim
        """
        newy = (
            torch.mm(torch.mean(vtab, dim=2).T, self.V0)[:, self.direction]
            / self.Nv ** 0.5
        )
        grad_pot = newy.T - self.w_hat_b
        square = torch.zeros(2, self.nb_point_dim[0], self.nb_point_dim[1])
        self.w_hat_tmp = np.zeros((2, self.nb_point_dim[0], self.nb_point_dim[1]))
        for i in range(0, grad_pot.shape[1], self.nb_point_dim[0]):
            self.w_hat_tmp[:, :, int(i / self.nb_point_dim[0])] = (
                self.w_hat_b[:, i : (i + self.nb_point_dim[0])].cpu().numpy()
            )
            square[:, :, int(i / self.nb_point_dim[0])] = grad_pot[
                :, i : (i + self.nb_point_dim[0])
            ]

        w_hat_dim = []
        for i in range(self.nDim):
            w_hat_dim.append(
                np.linspace(self.limits[0, i], self.limits[1, i], self.nb_point_dim[i])
            )

        res_x = np.zeros(self.nb_point_dim[0])
        for i in range(self.nb_point_dim[0]):
            res_x[i] = simps(
                square[0][: (i + 1), 0].cpu().numpy(), self.w_hat_tmp[0, : (i + 1), 0]
            )
        res_y = np.zeros((self.nb_point_dim[0], self.nb_point_dim[1]))
        for i in range(self.nb_point_dim[0]):
            for j in range(self.nb_point_dim[1]):
                res_y[i, j] = simps(
                    square[1][i, : (j + 1)].cpu().numpy(),
                    self.w_hat_tmp[1, i, : (j + 1)],
                )

        pot = np.expand_dims(res_x, 1).repeat(self.nb_point_dim[1].cpu(), 1) + res_y
        res = np.exp(self.N * (pot - np.max(pot)))

        const = np.zeros(res.shape[0])
        for i in range(res.shape[0]):
            const[i] = simps(res[:, i], self.w_hat_tmp[1, i, :])
        const = simps(const, self.w_hat_tmp[0, :, 0])
        self.p_m = torch.tensor(res / const, device=self.device, dtype=self.dtype)
        return square, self.p_m, self.w_hat_tmp, grad_pot, w_hat_dim

    def SampleTMC1D(
        self, n_sample: int, p_m: torch.Tensor, w_hat_b: torch.Tensor, region=None
    ):
        """
        Sample the constraint from the reconstructed probability distribution
        
        p_m: the reconstructed distribution 
        
        w_hat_b: the discretization

        n_sample: the number of samples to be generated

        region: the region of the distribution to be sampled
            
        """
        w_hat_b = w_hat_b.cpu().numpy()
        p_m = p_m.cpu().numpy()
        if region == None:
            region = torch.zeros(2)
            region[0] = w_hat_b.min()
            region[1] = w_hat_b.max()
        cdf = np.zeros(len(p_m) - 1)
        for i in range(1, len(w_hat_b)):
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

    def SampleTMC2D(self, n_sample, region=None):

        if region == None:
            region = torch.zeros(2, 2)
            region[0, 0] = self.w_hat_tmp[0].min()
            region[0, 1] = self.w_hat_tmp[0].max()
            region[1, 0] = self.w_hat_tmp[1].min()
            region[1, 1] = self.w_hat_tmp[1].max()

        p_y = np.zeros(self.p_m.shape[0])
        for i in range(1, len(p_y)):
            p_y[i - 1] = simps(
                torch.as_tensor(self.p_m[:, i]).cpu().numpy(),
                torch.as_tensor(self.w_hat_tmp[0, :, i]).cpu().numpy(),
            )
        sample_y = self.SampleTMC1D(
            n_sample,
            torch.as_tensor(p_y),
            torch.as_tensor(self.w_hat_tmp[1, 0, :]),
            region=region[1],
        )
        sample_x = []
        for i in range(len(sample_y)):
            id_y = (torch.tensor(self.w_hat_tmp[1, 0, :]) >= sample_y[i]).nonzero(
                as_tuple=True
            )[0][0]
            # print(id_y)
            sample_x.append(
                self.SampleTMC1D(
                    1,
                    torch.as_tensor(self.p_m[:, id_y] / p_y[id_y - 1]),
                    torch.as_tensor(self.w_hat_tmp[0, :, 1]),
                    region=region[0],
                )[0]
            )
            # print(id_y,' ',sample_x[-1])
        return torch.stack(sample_x).reshape(len(sample_x)), sample_y

    def genDataTMC2D(self, n_sample, V, it_mcmc, region=None):
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
        x_grid, y_grid = self.SampleTMC2D(n_sample, region=region)
        w_hat_b = torch.tensor(
            [x_grid.numpy(), y_grid.numpy()], device=self.device, dtype=self.dtype
        )
        vinit = torch.bernoulli(
            torch.rand((self.Nv, n_sample), device=self.device, dtype=self.dtype)
        )
        n_chain = self.nb_chain
        n_point = self.nb_point
        self.nb_chain = 1
        self.nb_point = n_sample
        self.direction = self.direction.to(torch.long)
        tmpv, _, _ = self.TMCSample(vinit, w_hat_b.cuda(), self.N, V, it_mcmc=it_mcmc)
        self.nb_chain = n_chain
        self.nb_point = n_point
        tmpv = tmpv.reshape(self.Nv, n_sample)
        si, mi, _, _, = self.Sampling(tmpv, it_mcmc=1)
        return si, mi

    def fit_batch(self, X: torch.Tensor):
        """
        Fit the model to the minibatch

        Parameters:
        -----------

        X : torch.tensor of shape (Nv, mb_s)
            Minibatch
        """
        h_pos_v, h_pos_m = self.SampleHiddens01(X)
        ## Weights SVD
        if not self.PCA:
            _, _, self.V0 = torch.svd(self.W)
            if torch.mean(self.V0[:, 0]) < 0:
                self.V0 = -self.V0

        ## Discretization
        proj_data = torch.mm(X.T, self.V0) / self.Nv ** 0.5
        self.limits = torch.zeros((2, self.nDim))
        for i in range(self.direction.shape[0]):
            self.limits[0, i] = (
                proj_data[:, self.direction[i]].min() - self.border_length
            )
            self.limits[1, i] = (
                proj_data[:, self.direction[i]].max() + self.border_length
            )
        x_grid = np.linspace(self.limits[0, 0], self.limits[1, 0], self.nb_point_dim[0])
        x_grid = np.array([x_grid for i in range(self.nb_point_dim[1])])
        x_grid = x_grid.reshape(self.nb_point)
        y_grid = []
        y_d = np.linspace(self.limits[0, 1], self.limits[1, 1], self.nb_point_dim[1])
        for i in range(self.nb_point_dim[1]):
            for j in range(self.nb_point_dim[0]):
                y_grid.append(y_d[i])
        self.w_hat_b = torch.tensor(
            [x_grid, y_grid], device=self.device, dtype=self.dtype
        )

        w_hat = torch.zeros(
            (self.nDim, self.nb_chain * self.nb_point), device=self.device
        )
        for i in range(self.nb_point):
            for j in range(self.nb_chain):
                w_hat[:, i * self.nb_chain + j] = self.w_hat_b[:, i]
        ## TMC Sampling
        self.X_pc, tmph, vtab = self.TMCSample(
            self.X_pc,
            w_hat,
            self.N,
            self.V0,
            it_mcmc=self.gibbs_steps,
            it_mean=self.it_mean,
        )
        ## Probability reconstruction

        (
            square,
            self.p_m,
            self.w_hat_tmp,
            grad_pot,
            w_hat_dim,
        ) = self.compute_probability(vtab)
        ## Observables reconstruction
        s_i = torch.mean(self.X_pc, dim=2)  # mean over nb_chain
        tau_a = torch.mean(tmph, dim=2)  # mean over nb_chain
        s_i_square = torch.zeros(
            [s_i.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]]
        )
        tau_a_square = torch.zeros(
            [tau_a.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]]
        )

        for i in range(0, grad_pot.shape[1], self.nb_point_dim[0]):
            s_i_square[:, :, int(i / self.nb_point_dim[0])] = s_i[
                :, i : (i + self.nb_point_dim[0])
            ]
            tau_a_square[:, :, int(i / self.nb_point_dim[0])] = tau_a[
                :, i : (i + self.nb_point_dim[0])
            ]

        prod = torch.zeros((self.Nv, self.Nh, self.nb_point), device=self.device)
        tmpcompute = torch.zeros(self.Nv, self.Nh, self.nb_chain)
        for i in range(self.nb_point):
            for k in range(self.nb_chain):
                tmpcompute[:, :, k] = torch.outer(self.X_pc[:, i, k], tmph[:, i, k])
            prod[:, :, i] = torch.mean(tmpcompute, dim=2)

        s_i_square = torch.zeros(
            [s_i.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]],
            device=self.device,
            dtype=self.dtype,
        )
        tau_a_square = torch.zeros(
            [tau_a.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]],
            device=self.device,
            dtype=self.dtype,
        )
        prod_square = torch.zeros(
            (prod.shape[0], prod.shape[1], self.nb_point_dim[0], self.nb_point_dim[1]),
            device=self.device,
            dtype=self.dtype,
        )
        for i in range(0, grad_pot.shape[1], self.nb_point_dim[0]):
            s_i_square[:, :, int(i / self.nb_point_dim[0])] = s_i[
                :, i : (i + self.nb_point_dim[0])
            ]
            tau_a_square[:, :, int(i / self.nb_point_dim[0])] = tau_a[
                :, i : (i + self.nb_point_dim[0])
            ]
            prod_square[:, :, :, int(i / self.nb_point_dim[0])] = prod[
                :, :, i : (i + self.nb_point_dim[0])
            ]

        tmpres_s_i = torch.zeros(
            self.Nv, self.nb_point_dim[0], device=self.device, dtype=self.dtype
        )
        tmpres_tau_a = torch.zeros(
            self.Nh, self.nb_point_dim[0], device=self.device, dtype=self.dtype
        )
        tmpres_prod = torch.zeros(
            (prod_square.shape[0], prod_square.shape[1], prod_square.shape[2]),
            device=self.device,
            dtype=self.dtype,
        )
        s_i_square = self.p_m * s_i_square  # Ca fait bien ce qu'on veut
        tau_a_square = self.p_m * tau_a_square
        prod_square = self.p_m * prod_square
        s_i_fin = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
        tau_a_fin = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        prod_fin = torch.zeros(
            prod_square.shape[0],
            prod_square.shape[1],
            device=self.device,
            dtype=self.dtype,
        )
        for i in range(self.nb_point_dim[0]):
            tmpres_s_i[:, i] = torch.trapz(
                s_i_square[:, i, :],
                torch.tensor(w_hat_dim[1], device=self.device, dtype=self.dtype),
            )
            tmpres_tau_a[:, i] = torch.trapz(
                tau_a_square[:, i, :],
                torch.tensor(w_hat_dim[1], device=self.device, dtype=self.dtype),
            )
            tmpres_prod[:, :, i] = torch.trapz(
                prod_square[:, :, i, :],
                torch.tensor(w_hat_dim[1], device=self.device, dtype=self.dtype),
            )
        tau_a_fin = torch.trapz(
            tmpres_tau_a,
            torch.tensor(w_hat_dim[0], device=self.device, dtype=self.dtype),
        )
        s_i_fin = torch.trapz(
            tmpres_s_i, torch.tensor(w_hat_dim[0], device=self.device, dtype=self.dtype)
        )
        prod_fin = torch.trapz(
            tmpres_prod,
            torch.tensor(w_hat_dim[0], device=self.device, dtype=self.dtype),
        )

        self.X_pc = self.X_pc.reshape(self.Nv, self.nb_point * self.nb_chain)

        ## Update weights
        if self.UpdCentered:
            self.updateWeightsCentered(
                X, h_pos_v, h_pos_m, s_i_fin, tau_a_fin, prod_fin.T
            )
        else:
            self.updateWeights(X, h_pos_m, s_i_fin, tau_a_fin, prod_fin.T)

    def fit(self, X: torch.Tensor, ep_max=0):
        """
        Fit the model to the dataset

        Parameters:
        -----------

        X : torch.tensor of shape (Nv, Ns)
            Training dataset

        ep_max : int, default=0
            number of training epochs. 
            If = 0, uses self.ep_max instead.
        """
        if self.PCA:
            _, _, self.V0 = torch.svd(X.T)
            self.V0 = self.V0.to(self.device)
            if torch.mean(self.V0[:, 0]) < 0:
                self.V0 = -self.V0

        if ep_max == 0:
            ep_max = self.ep_max

        NB = int(X.shape[1] / self.mb_s)

        if self.ep_tot == 0:
            self.VisDataAv = torch.mean(X, 1)

        if (len(self.list_save_time) > 0) & (self.up_tot == 0):
            f = h5py.File(MODEL_DIR.joinpath(f"AllParameters{self.file_stamp}.h5"), "w")
            f.create_dataset(f"alltime", data=self.list_save_time)
            f.close()

        if (len(self.list_save_rbm) > 0) & (self.ep_tot == 0):
            f = h5py.File(MODEL_DIR.joinpath(f"RBM{self.file_stamp}.h5"), "w")
            f.create_dataset(f"lr", data=self.lr)
            f.create_dataset(f"NGibbs", data=self.gibbs_steps)
            f.create_dataset(f"UpdByEpoch", data=NB)
            f.create_dataset(f"miniBatchSize", data=self.mb_s)
            f.create_dataset(f"numPCD", data=self.num_pcd)
            f.create_dataset(f"alltime", data=self.list_save_rbm)
            f.close()

        for t in range(ep_max):
            print("IT ", self.ep_tot)
            self.ep_tot += 1

            Xp = X[:, torch.randperm(X.size()[1])]

            for m in range(NB):
                if self.ResetPermChainBatch:
                    self.X_pc = torch.bernoulli(
                        torch.rand(
                            self.Nv, self.nb_chain * self.nb_point, device=self.device
                        )
                    )

                Xb = self.getMiniBatches(Xp, m)
                self.fit_batch(Xb)

                if self.up_tot in self.list_save_time:
                    f = h5py.File(
                        MODEL_DIR.joinpath(f"AllParameters{self.file_stamp}.h5"), "a"
                    )
                    print(f"Saving nb_upd=" + str(self.up_tot))
                    f.create_dataset(f"W{self.up_tot}", data=self.W.cpu())
                    f.create_dataset(f"vbias{self.up_tot}", data=self.vbias.cpu())
                    f.create_dataset(f"hbias{self.up_tot}", data=self.hbias.cpu())
                    f.create_dataset(f"p_m{self.up_tot}", data=self.p_m.cpu())
                    f.create_dataset(f"w_hat{self.up_tot}", data=self.w_hat_tmp)
                    f.close()

                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                f = h5py.File(MODEL_DIR.joinpath(f"RBM{self.file_stamp}.h5"), "a")
                f.create_dataset(f"W{self.ep_tot}", data=self.W.cpu())
                f.create_dataset(f"vbias{self.ep_tot}", data=self.vbias.cpu())
                f.create_dataset(f"hbias{self.ep_tot}", data=self.hbias.cpu())
                f.create_dataset(f"p_m{self.ep_tot}", data=self.p_m.cpu())
                f.create_dataset(f"w_hat{self.ep_tot}", data=self.w_hat_tmp)
                if self.ep_tot in self.list_save_xpc:
                    f.create_dataset(f"X_pc{self.ep_tot}", data=self.X_pc.cpu())
                f.close()

        print(
            f"model updates saved at {MODEL_DIR.joinpath(f'AllParameters{self.file_stamp}.h5')}"
        )
        print(f"model saved at {MODEL_DIR.joinpath(f'RBM{self.file_stamp}.h5')}")

