import torch
import numpy as np
import h5py
from scipy.integrate import simps

from rbm.definitions import MODEL_DIR
from rbm.models import RBM


class TMCRBM(RBM):
    """
    PyTorch implementation of the Restricted Boltzmann Machine using the Tethered Monte Carlo Algorithm

    Parameters:
    -----------

    num_visible : int
        Number of visible units

    num_hidden : int
        Number of hidden units
    
    device : torch.device
        Wether to use CPU or GPU, should be the same as the dataset

    gibbs_step : int, default=10
        number if MCMC steps for computing the negative term

    var_init : float, default=1e-4
        Variance of the initialised weights

    dtype : torch.dtype, default=torch.float
        Datatype of the model, should be the same as the dataset

    num_pcd : int, default=100
        Number of permanent chains

    lr : float, default=0.01
        Learning rate

    ep_max : int, default=100
        Number of training epochs

    mb_s : int, default=50
        Minibatch size
    
    UpdCentered : bool, default=False
        Update using centered gradients

    CDLearning : bool, default=False
        Train using Contrastive Divergence

    UpdFieldsVis : bool, default=True
        Update visible fields

    UpdFieldsHid : bool, default=True
        Update hidden fields

    UpdWeights : bool, defaukt=True
        Update the weights

    it_mean : int, default=8
        Number of iterations used for temporal mean
    
    nb_chain : int, default=15
        number of chain for each discretization point

    N : int, default=20000
        Constraint on gaussian bath
    
    nb_point : int, default=1000
        Number of points used for discretization
    
    border_length : float, default=0.2
        Length around the data used to compute the probability
    
    direction : int, default=0
        The direction used for the discretization
    
    PCA : bool, default=False
        Use PCA for projection along the direction

    Attributes:
    -----------

    Nv : int
        Number of visible units

    Nh : int
        Number of hidden units

    W : torch.tensor of shape (Nh, Nv)
        Weight matrix of the RBM

    vbias : torch.tensor of shape (Nv)
        Bias on the visible units

    hbias : torch.tensor of shape (Nh)
        Bias on the hidden units

    device : torch.device
        Device used

    gibbs_step : int
        number if MCMC steps for computing the negative term

    var_init : float
        Variance of the initialised weights

    dtype : torch.dtype
        Datatype of the model, should be the same as the dataset

    num_pcd : int
        Number of permanent chains

    X_pc : torch.tensor of shape (Nv, num_pcd)
        Permanent chains for PCD

    lr : float
        Learning rate

    ep_max : int
        Number of training epochs

    mb_s : int
        Minibatch size
    
    UpdCentered : bool
        Update using centered gradients

    CDLearning : bool
        Train using Contrastive Divergence

    ep_tot : int
        Total number of epoch trained

    up_tot : int
        Total number of gradient update performed
    
    list_save_time : list
        Timecode at which the RBM is saved
    
    list_save_rbm : list
    
    file_stamp : str
        filename stamp used to save figures and model
    
    VisDataAv : float
        Average value of the visible units
    
    HidDataAv : float
        Average value of the hidden units
        
    ResetPermChainBatch : bool
        Reset the permanent chains at the beginning of each batch
    
    UpdFieldsVis    : bool

    UpdFieldsHid : bool
    
    UpdWeights : bool
    
    UpdWα : bool

    u : torch.tensor
    
    s : torch.tensor
    
    v : torch.tensor

    nb_chain : int  

    it_mean : int  

    N : int

    nb_point : int

    border_length : int

    save_fig : bool

    direction : int

    PCA : bool

    p_m : torch.tensor of shape (nb_point-1)
        Probability on the discretized space

    Ω = torch.tensor of shape (nb_point-1) 
        Potential on the discretized space
    
    w_hat_b : torch.tensor of shape (nb_point)
        Discretization points along the direction

    V0 : torch.tensor of shape (1 #TODO)
        Projection vector

    ParalelleTMC : bool
        Use parallel TMC algorithm (NOT WORKING)
    
    """

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
        UpdFieldsVis=True,
        UpdFieldsHid=True,
        UpdWeights=True,
        it_mean=8,
        nb_chain=15,
        N=20000,
        nb_point=1000,
        border_length=0.2,
        direction=0,
        PCA=False,
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
        self.ResetPermChainBatch = ResetPermChainBatch

        # Adjusting which parameter we want to learn!
        self.UpdFieldsVis = UpdFieldsVis
        self.UpdFieldsHid = UpdFieldsHid
        self.UpdWeights = UpdWeights
        self.UpdWα = False

        self.u = torch.tensor([])
        self.s = torch.tensor([])
        self.v = torch.tensor([])

        # TMC param
        self.nb_chain = nb_chain
        self.it_mean = it_mean
        self.N = N
        self.nb_point = nb_point
        self.border_length = border_length
        self.save_fig = save_fig
        self.direction = direction
        self.PCA = PCA

        self.p_m = torch.zeros(nb_point - 1)  # Probabilité en fonction de la contrainte
        self.Ω = torch.zeros(nb_point - 1)  # Potential
        self.w_hat_b = torch.zeros(nb_point)  # discrétisation de la direction
        self.V0 = torch.zeros(1, device=self.device)  # Vecteur de projection
        self.ParalelleTMC = False
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
        Tethered Monte-Carlo algorithm on the visible units

        Parameters:
        -----------
        v : torch.tensor of shape (Nv, nb_point*nb_chain)
            starting point for the Markov chains

        w_hat : torch.tensor of shape (nb_point*nb_chain)
            Constraint parameter values

        N : int
            Constraint on the gaussian bath
        
        V : torch.tensor of shape (Nv)
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

        v_curr : torch.tensor of shape (Nv, nb_point*nb_chain)
            The last visible value of the MC chains

        h_curr : torch.tensor of shape (Nh, nb_point*nb_chain)
            The last hidden value of the MC chains

        vtab : torch.tensor of shape (Nv, nb_point*nb_chain)
            The temporal mean over the last it_mean visible values for each MC chain
        """
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps
        if it_mean == 0:
            it_mean = self.it_mean

        vtab = torch.zeros(v.shape, device=self.device)
        v_curr = v

        norm = 1 / (v_curr.shape[0] ** 0.5)
        w_curr = torch.mv(v_curr.T, V) * norm  # Compute current tethered weight

        # we perform it_mcmc steps de MC
        for t in range(it_mcmc):
            h_curr, _ = self.SampleHiddens01(v_curr, β=β)
            h_i = torch.mm(self.W.T, h_curr) + self.vbias.reshape(
                v.shape[0], 1
            )  # Nv x Ns
            w_next = w_curr.clone()
            v_next = torch.clone(v_curr)

            # we visit all sites to propose MC flip
            Pe = torch.randperm(self.Nv)
            for idx in range(self.Nv):
                i = Pe[idx]

                v_next[i, :] = 1 - v_curr[i, :]  # proposed change
                w_next += (
                    (2 * v_next[i, :] - 1) * V[i] * norm
                )  # Compute new tethered weight

                # Compute -DeltaE
                ΔE = β * ((2 * v_next[i, :] - 1) * h_i[i, :]) - (N / 2) * (
                    (w_hat - w_next) ** 2 - (w_hat - w_curr) ** 2
                )
                tir = torch.rand(v_curr.shape[1], 1, device=self.device).squeeze()
                prob = torch.exp(ΔE).squeeze()

                v_curr[i, :] = torch.where(tir < prob, v_next[i, :], v_curr[i, :])
                v_next[i, :] = torch.where(tir < prob, v_next[i, :], 1 - v_next[i, :])
                w_curr = torch.where(tir < prob, w_next, w_curr)
                w_next = torch.where(tir < prob, w_next, w_curr)

            # Temporal mean over the last it_mean iterations
            if t >= (it_mcmc - it_mean):
                vtab += v_curr
        vtab = vtab * (1 / it_mean)
        return v_curr, h_curr, vtab

    def TMCSampleHidd(
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
        Tethered Monte-Carlo algorithm on the hidden units

        Parameters:
        -----------
        v : torch.tensor of shape (Nv, nb_point*nb_chain)
            starting point for the Markov chains

        w_hat : torch.tensor of shape (nb_point*nb_chain)
            Constraint parameter values

        N : int
            Constraint on the gaussian bath
        
        V : torch.tensor of shape (Nv)
            The projection vector from the dataset space to the constrained dimension
        
        it_mcmc : 
            The number of MCMC steps.
            If set to 0, use the class variable self.it_mcmc instead.
        
        it_mean : int, default=0
            The number of MCMC steps used for the temporal mean.
            If set to 0, use the class variable self.it_mean instead.
        
        β : float, default=1
            effective temperature. Used for annealing
        
        Returns:
        --------

        v_curr : torch.tensor of shape (Nv, nb_point*nb_chain)
            The last visible value of the MC chains

        h_curr : torch.tensor of shape (Nh, nb_point*nb_chain)
            The last hidden value of the MC chains

        htab : torch.tensor of shape (Nh, nb_point*nb_chain)
            The temporal mean over the last it_mean hidden values for each MC chain
        """
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps
        if it_mean == 0:
            it_mean = self.it_mean

        v_curr, _, h_curr, _ = self.Sampling(v)
        htab = torch.zeros(h_curr.shape, device=self.device)

        norm = 1 / (h_curr.shape[0] ** 0.5)
        w_curr = torch.mv(h_curr.T, V) * norm  # Compute current tethered weight

        # we perform it_mcmc steps de MC
        for t in range(it_mcmc):

            h_next = torch.clone(h_curr)
            w_next = w_curr.clone()

            h_i = torch.mm(self.W, v_curr) + self.hbias.reshape(
                h_curr.shape[0], 1
            )  # Nh x Ns

            Pe = torch.randperm(self.Nh)
            for idx in range(self.Nh):
                i = Pe[idx]
                h_next[i, :] = 1 - h_curr[i, :]  # proposed change
                w_next += (
                    (2 * h_next[i, :] - 1) * V[i] * norm
                )  # Compute new tethered weight

                # Compute -DeltaE
                ΔE = β * ((2 * h_next[i, :] - 1) * h_i[i, :]) - (N / 2) * (
                    (w_hat - w_next) ** 2 - (w_hat - w_curr) ** 2
                )
                tir = torch.rand(h_curr.shape[1], 1, device=self.device).squeeze()
                prob = torch.exp(ΔE).squeeze()

                # Update chains position with probability prob
                h_curr[i, :] = torch.where(tir < prob, h_next[i, :], h_curr[i, :])
                h_next[i, :] = torch.where(tir < prob, h_next[i, :], 1 - h_next[i, :])
                w_curr = torch.where(tir < prob, w_next, w_curr)
                w_next = torch.where(tir < prob, w_next, w_curr)

            v_curr, _ = self.SampleVisibles01(h_curr, β=β)

            # Temporal mean over the last it_mean iterations
            if t >= (it_mcmc - it_mean):
                htab += h_curr
        htab = htab * (1 / it_mean)
        return v_curr, h_curr, htab

    def updateWeightsCentered(
        self,
        v_pos: torch.Tensor,
        h_pos_m: torch.Tensor,
        v_neg: torch.Tensor,
        h_neg_v: torch.Tensor,
        h_neg_m: torch.Tensor,
    ):
        """
        
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
        ΔW = (Hc_pos.mm(Xc_pos.t()) * NormPos - ΔW_neg).float()

        if self.UpdWeights:
            self.W += ΔW * self.lr

        if self.UpdWα:
            if self.u.shape[0] == 0:
                self.u, self.s, self.v = torch.svd(self.W)

            # ΔW = h_pos.mm(v_pos.t())/self.mb_s - h_neg.mm(v_neg.t())/self.num_pcd
            self.s += self.lr * torch.diag(
                torch.mm(torch.mm(self.u.t(), ΔW), self.v)
            )  # - 2*self.lr*self.regL2*self.s
            self.W = torch.mm(torch.mm(self.u, torch.diag(self.s)), self.v.t())

        if self.UpdFieldsVis:
            ΔVB = (
                torch.sum(v_pos, 1) * NormPos
                - si_neg
                - torch.mv(ΔW.t().float(), self.HidDataAv)
            )
            self.vbias += self.lr * ΔVB

        if self.UpdFieldsHid:
            ΔHB = (
                torch.sum(h_pos_m, 1) * NormPos
                - τa_neg
                - torch.mv(ΔW.float(), self.VisDataAv)
            )
            self.hbias += self.lr * ΔHB

    def compute_probability(self, vtab: torch.Tensor):
        """
        Computes the probability of the constrained dimension

        Parameters:
        -----------
        vtab : Tensor of shape (Nh, nb_chain*nb_point)
            The temporal mean over the last it_mean visible values for each MC chain

        Returns:
        --------
        res : The discretization along the constrained dimension

        p_m : The value of the probability at each point of the discretization
        """
        y = (
            np.array(torch.mm(vtab.T, self.V0.unsqueeze(1)).cpu().squeeze())
            / self.Nv ** 0.5
        )  # Projection sur le paramètre de contrainte
        newy = np.mean(y.reshape(self.nb_point, self.nb_chain), 1)
        # newy = np.array([np.mean(y[i*self.nb_chain:i*self.nb_chain+self.nb_chain])
        #                    for i in range(self.nb_point)]) # mean over nb_chain
        w_hat_b_np = self.w_hat_b.cpu().numpy()
        res = np.zeros(len(self.w_hat_b) - 1)
        # integrale sur la dérivée
        for i in range(1, len(self.w_hat_b)):
            res[i - 1] = simps(newy[:i] - w_hat_b_np[:i], w_hat_b_np[:i])
        # constante de normalisation
        const = simps(np.exp(self.N * res - np.max(self.N * res)), w_hat_b_np[:-1])
        grad_pot = newy - w_hat_b_np
        return (
            res,
            torch.tensor(
                np.exp(self.N * res - np.max(self.N * res)) / const, device=self.device
            ),
            grad_pot,
            w_hat_b_np,
        )

    def computeProbabilityHidd(self, htab: torch.Tensor):
        """
        
        """
        y = (
            np.array(torch.mm(htab.T, self.U0.unsqueeze(1)).cpu().squeeze())
            / self.Nh ** 0.5
        )  # Projection sur le paramètre de contrainte
        newy = np.array(
            [
                np.mean(y[i * self.nb_chain : i * self.nb_chain + self.nb_chain])
                for i in range(self.nb_point)
            ]
        )  # mean over nb_chain
        w_hat_b_np = self.w_hat_b.cpu().numpy()
        res = np.zeros(len(self.w_hat_b) - 1)
        # integrale sur la dérivée
        for i in range(1, len(self.w_hat_b)):
            res[i - 1] = simps(newy[:i] - w_hat_b_np[:i], w_hat_b_np[:i])
        # constante de normalisation
        const = simps(np.exp(self.N * res - np.max(self.N * res)), w_hat_b_np[:-1])
        return (
            res,
            torch.tensor(
                np.exp(self.N * res - np.max(self.N * res)) / const, device=self.device
            ),
        )

    def computeProbabilityAnalysis(self, X: torch.Tensor):
        start = torch.bernoulli(
            torch.rand(self.Nv, self.nb_chain * self.nb_point, device=self.device)
        )
        # PCA or weigths SVD
        if self.PCA:
            _, _, self.V0 = torch.svd(X.T)
            self.V0 = self.V0[:, self.direction].to(self.device)
            if torch.mean(self.V0) < 0:
                self.V0 = -self.V0
        else:
            _, _, self.V0 = torch.svd(self.W)
            self.V0 = self.V0[:, self.direction]
            if torch.mean(self.V0) < 0:
                self.V0 = -self.V0

        # discretization
        proj_data = torch.mv(X.T, self.V0) / (self.Nv ** 0.5)
        xmin = torch.min(proj_data) - self.border_length
        xmax = torch.max(proj_data) + self.border_length

        self.w_hat_b = torch.linspace(
            xmin, xmax, steps=self.nb_point, device=self.device
        )
        # discretization by repeating nb_chain times each point
        w_hat = torch.zeros(self.nb_chain * self.nb_point, device=self.device)
        for i in range(self.nb_point):
            for j in range(self.nb_chain):
                w_hat[i * self.nb_chain + j] = self.w_hat_b[i]
        print("Sampling!")  # TMC Sampling
        vtab = 0
        if self.ParalelleTMC:
            _, _, vtab = self.TMCSampleParallel(
                start,
                w_hat,
                self.N,
                self.V0,
                it_mcmc=self.gibbs_steps,
                it_mean=self.it_mean,
            )
        else:
            _, _, vtab = self.TMCSample(
                start,
                w_hat,
                self.N,
                self.V0,
                it_mcmc=self.gibbs_steps,
                it_mean=self.it_mean,
            )
        print("Recstr Proba!")  # Probability reconstruction
        self.Ω, self.p_m, _, _ = self.compute_probability(vtab)
        return self.Ω

    def computeProbabilityAnalysisHidd(self, X):
        start = torch.bernoulli(
            torch.rand(self.Nv, self.nb_chain * self.nb_point, device=self.device)
        )
        # PCA or weigths SVD

        self.U0, _, _ = torch.svd(self.W)
        self.U0 = self.U0[:, self.direction]
        if torch.mean(self.U0) < 0:
            self.U0 = -self.U0

        # discretization
        _, _, _, H = self.Sampling(X)
        proj_data = torch.mv(H.T, self.U0) / (self.Nh ** 0.5)
        xmin = torch.min(proj_data) - self.border_length
        xmax = torch.max(proj_data) + self.border_length

        self.w_hat_b = torch.linspace(
            xmin, xmax, steps=self.nb_point, device=self.device
        )
        # discretization by repeating nb_chain times each point
        w_hat = torch.zeros(self.nb_chain * self.nb_point, device=self.device)
        for i in range(self.nb_point):
            for j in range(self.nb_chain):
                w_hat[i * self.nb_chain + j] = self.w_hat_b[i]
        print("Sampling!")  # TMC Sampling
        vtab = 0

        _, hc, htab = self.TMCSampleHidd(
            start,
            w_hat,
            self.N,
            self.U0,
            it_mcmc=self.gibbs_steps,
            it_mean=self.it_mean,
        )
        print("Recstr Proba!")  # Probability reconstruction
        self.Ω, self.p_m = self.computeProbabilityHidd(htab)
        return self.Ω, hc

    def SampleTMC1D(self, n_sample: int, region=None):
        """
        Sample the constraint from the reconstructed probability distribution
        
        p_m: the reconstructed distribution 
        
        w_hat_b: the discretization

        n_sample: the number of samples to be generated

        region: the region of the distribution to be sampled
            
        """
        p_m = self.p_m.cpu().numpy()
        w_hat_b = self.w_hat_b.cpu().numpy()
        if region == None:
            region = torch.zeros(2)
            region[0] = self.w_hat_b.min()
            region[1] = self.w_hat_b.max()
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
                    if self.w_hat_b[k] <= region[1] and self.w_hat_b[k] >= region[0]:
                        sample[i] = self.w_hat_b[k]
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

    def genDataTMC1D(self, n_sample, V, it_mcmc=30, region=None):
        """
        Generate data from the TMCRBM

        n_sample: int
            The number of samples to be generated

        V: Tensor of shape (Nv)
            The projection vector from the dataset space to the constrained dimension

        it_mcmc: int, default=30
            The number of iterations of the mcmc algorithm

        region: Tensor of shape (2), default=None
            The region of the distribution to be sampled

        """
        gen_m = self.SampleTMC1D(n_sample, region=region)
        vinit = torch.bernoulli(
            torch.rand((self.Nv, n_sample), device=self.device, dtype=self.dtype)
        )
        tmpv, _, _ = self.TMCSample(vinit, gen_m.cuda(), self.N, V, it_mcmc=it_mcmc)
        si, mi, _, _ = self.Sampling(tmpv, it_mcmc=1)
        return si, mi

    def fit_batch(self, X):
        h_pos_v, h_pos_m = self.SampleHiddens01(X)
        # PCA or weigths SVD
        if not self.PCA:
            _, _, self.V0 = torch.svd(self.W)
            self.V0 = self.V0[:, self.direction]
            if torch.mean(self.V0) < 0:
                self.V0 = -self.V0

        # discretization
        proj_data = torch.mv(X.T, self.V0) / (self.Nv ** 0.5)
        xmin = torch.min(proj_data) - self.border_length
        xmax = torch.max(proj_data) + self.border_length
        self.w_hat_b = torch.linspace(
            xmin, xmax, steps=self.nb_point, device=self.device
        )
        # discretization by repeating nb_chain times each point
        w_hat = torch.zeros(self.nb_chain * self.nb_point, device=self.device)
        for i in range(self.nb_point):
            for j in range(self.nb_chain):
                w_hat[i * self.nb_chain + j] = self.w_hat_b[i]
        # A AMELIORER

        # TMC Sampling
        self.X_pc, biased_hid, vis_time_av = self.TMCSample(
            self.X_pc,
            w_hat,
            self.N,
            self.V0,
            it_mcmc=self.gibbs_steps,
            it_mean=self.it_mean,
        )
        # Probability reconstruction
        self.Ω, self.p_m, _, _ = self.compute_probability(vis_time_av)

        Ns = self.X_pc.shape[1]
        # Observable reconstruction
        s_i = torch.mean(self.X_pc.reshape(self.Nv, self.nb_point, self.nb_chain), 2)
        tau_a = torch.mean(biased_hid.reshape(self.Nh, self.nb_point, self.nb_chain), 2)
        s_i = torch.trapz(s_i[:, 1:] * self.p_m, self.w_hat_b[1:], dim=1)
        tau_a = torch.trapz(tau_a[:, 1:] * self.p_m, self.w_hat_b[1:], dim=1)

        prod = torch.matmul(
            self.X_pc.T.reshape(Ns, self.Nv, 1), biased_hid.T.reshape(Ns, 1, self.Nh)
        )
        prod = prod.transpose(0, 1).transpose(1, 2)
        prod = torch.mean(
            prod.reshape(self.Nv, self.Nh, self.nb_point, self.nb_chain), 3
        )
        prod = torch.trapz(prod[:, :, 1:] * self.p_m, self.w_hat_b[1:], dim=2)

        if self.UpdCentered:
            self.updateWeightsCentered(X, h_pos_m, s_i, tau_a, prod.T)
        else:
            self.updateWeights(X, h_pos_m, s_i, tau_a, prod.T)

    def fit(self, X: torch.Tensor, ep_max=0):

        if self.PCA:
            _, _, self.V0 = torch.svd(X.T)
            self.V0 = self.V0[:, self.direction].to(self.device)
            if torch.mean(self.V0) < 0:
                self.V0 = -self.V0

        if ep_max == 0:
            ep_max = self.ep_max

        NB = int(X.shape[1] / self.mb_s)

        if self.ep_tot == 0:
            self.VisDataAv = torch.mean(X, 1)

        if (len(self.list_save_time) > 0) & (self.up_tot == 0):
            f = h5py.File(MODEL_DIR.joinpath(f"AllParameters{self.file_stamp}.h5"), "w")
            f.create_dataset("alltime", data=self.list_save_time)
            f.close()

        if (len(self.list_save_rbm) > 0) & (self.ep_tot == 0):
            f = h5py.File(MODEL_DIR.joinpath(f"RBM{self.file_stamp}.h5"), "w")
            f.create_dataset("lr", data=self.lr)
            f.create_dataset("NGibbs", data=self.gibbs_steps)
            f.create_dataset("UpdByEpoch", data=NB)
            f.create_dataset("miniBatchSize", data=self.mb_s)
            f.create_dataset("numPCD", data=self.num_pcd)
            f.create_dataset("alltime", data=self.list_save_rbm)
            f.close()

        for t in range(ep_max):
            print(f"IT {self.ep_tot}")
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
                    print(f"Saving nb_upd={self.up_tot}")
                    f.create_dataset(f"W{self.up_tot}", data=self.W.cpu())
                    f.create_dataset(f"vbias{self.up_tot}", data=self.vbias.cpu())
                    f.create_dataset(f"hbias{self.up_tot}", data=self.hbias.cpu())
                    f.create_dataset(f"p_m{self.up_tot}", data=self.p_m.cpu())
                    f.create_dataset(f"pot{self.up_tot}", data=self.Ω)
                    f.close()

                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                f = h5py.File(MODEL_DIR.joinpath(f"RBM{self.file_stamp}.h5"), "a")
                f.create_dataset(f"W{self.ep_tot}", data=self.W.cpu())
                f.create_dataset(f"vbias{self.ep_tot}", data=self.vbias.cpu())
                f.create_dataset(f"hbias{self.ep_tot}", data=self.hbias.cpu())
                f.create_dataset(f"p_m{self.ep_tot}", data=self.p_m.cpu())
                f.create_dataset(f"pot{self.ep_tot}", data=self.Ω)
                f.create_dataset(f"X_pc{self.up_tot}", data=self.X_pc.cpu())
                f.close()

        print(
            f"model updates saved at {MODEL_DIR.joinpath(f'AllParameters{self.file_stamp}.h5')}"
        )
        print(f"model saved at {MODEL_DIR.joinpath(f'RBM{self.file_stamp}.h5')}")

