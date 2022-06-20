import h5py
import numpy as np
import torch


class RBM:
    """
    PyTorch implementation of the Restricted Boltzmann Machine

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
        UpdCentered=False,
        CDLearning=False,
    ):
        self.Nv = num_visible
        self.Nh = num_hidden
        self.gibbs_steps = gibbs_steps
        self.device = device
        self.dtype = dtype
        self.W = (
            torch.randn(size=(self.Nh, self.Nv), device=self.device, dtype=self.dtype)
            * var_init
        )
        self.var_init = var_init
        self.vbias = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
        self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        self.X_pc = torch.bernoulli(
            torch.rand((self.Nv, num_pcd), device=self.device, dtype=self.dtype)
        )
        self.lr = lr
        self.ep_max = ep_max
        self.mb_s = mb_s
        self.num_pcd = num_pcd

        self.ep_tot = 0
        self.up_tot = 0
        self.list_save_time = []
        self.list_save_rbm = []
        self.file_stamp = ""
        self.VisDataAv = 0
        self.HidDataAv = 0
        self.UpdCentered = UpdCentered
        self.ResetPermChainBatch = True
        self.CDLearning = CDLearning

    def ImConcat(self, X: torch.Tensor, ncol=10, nrow=5, sx=28, sy=28, ch=1):
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

    def SetVisBias(self, X: torch.Tensor):
        """
        Initialise the visible bias using the empirical frequency of the training dataset

        Inputs :
        -----------

        X : torch.tensor of shape (Nv, Ns)
            The training dataset
        """
        NS = X.shape[1]
        prob1 = torch.sum(X, 1) / NS
        prob1 = torch.clamp(prob1, min=1e-5)
        prob1 = torch.clamp(prob1, max=1 - 1e-5)
        self.vbias = -torch.log(1.0 / prob1 - 1.0)

    def InitXpc(self, V: torch.Tensor):
        """
        Set the initial value for the permanent chain used for PCD

        Inputs :
        ------------

        V : torch.tensor of shape (Nv, self.num_pcd)
            The starting values for the PCD chains
        """

        self.X_pc = V

    def SampleHiddens01(self, V: torch.Tensor, β=1):
        """
        Sample the hidden units knowing the visible units values

        Inputs :
        -----------

        V : torch.tensor of shape (Nv, Ns)
            values of of the visible unit for each chain
        
        β : float, default=1
            effective temperature

        Returns :
        --------

        h : torch.tensor of shape (Nh, Ns)
            values of the hidden units for each chain

        mh : torch.tensor of shape (Nh, Ns)
            average values of the hidden units for each chain
        """

        mh = torch.sigmoid(β * (self.W.mm(V).t() + self.hbias).t())
        h = torch.bernoulli(mh)
        return h, mh

    def SampleVisibles01(self, H: torch.Tensor, β=1):
        """
        Sample the visible units knowing the hidden units values

        Parameters:
        -----------

        H : torch.tensor of shape (Nh, Ns)
            values of of the hidden unit for each chain
        
        β : float, default=1
            effective temperature

        Returns:
        --------

        v : torch.tensor of shape (Nv, Ns)
            values of the visible units for each chain

        mv : torch.tensor of shape (Nv, Ns)
            average values of the visible units for each chain
        """

        mv = torch.sigmoid(β * (self.W.t().mm(H).t() + self.vbias).t())
        v = torch.bernoulli(mv)
        return v, mv

    def GetAv(self, it_mcmc=0, β=1):
        """
        Compute the negative term for the gradient

        Parameters:
        -----------
        it_mcmc : int, default=0
            number of MCMC steps.  
            if it_mcmc == 0, use the class variables self.gibbs_steps instead

        β : float, default=1
            effective temperature. Used for annealing

        Returns:
        --------
        v : torch.tensor of shape (Nv, Ns)
            The last visible value of the MC chains
        
        mv : torch.tensor of shape (Nv, Ns)
            The last visible mean value of the MC chains

        h : torch.tensor of shape (Nh, Ns)
            The last hidden value of the MC chains

        mh : torch.tensor of shape (Nh, Ns)
            The last hidden mean value of the MC chains
        """

        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps

        v = self.X_pc
        mh = 0

        h, mh = self.SampleHiddens01(v, β=β)
        v, mv = self.SampleVisibles01(h, β=β)

        for _ in range(1, it_mcmc):
            h, mh = self.SampleHiddens01(v, β=β)
            v, mv = self.SampleVisibles01(h, β=β)

        return v, mv, h, mh

    def Sampling(self, X: torch.Tensor, it_mcmc=0, β=1):
        """
        Sample the RBM

        Parameters:
        -----------
        X : torch.tensor of shape (Nv, Ns)
            starting position for the MC chains

        it_mcmc : int, default=0
            number of MCMC steps.  
            if it_mcmc == 0, use the class variables self.gibbs_steps instead

        β : float, default=1
            effective temperature. Used for annealing

        Returns:
        --------
        v : torch.tensor of shape (Nv, Ns)
            The last visible value of the MC chains
        
        mv : torch.tensor of shape (Nv, Ns)
            The last visible mean value of the MC chains

        h : torch.tensor of shape (Nh, Ns)
            The last hidden value of the MC chains

        mh : torch.tensor of shape (Nh, Ns)
            The last hidden mean value of the MC chains

        """

        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps

        v = X
        β = 1

        h, mh = self.SampleHiddens01(v, β=β)
        v, mv = self.SampleVisibles01(h, β=β)

        h, mh = self.SampleHiddens01(v, β=β)
        v, mv = self.SampleVisibles01(h, β=β)

        for _ in range(it_mcmc - 1):
            h, mh = self.SampleHiddens01(v, β=β)
            v, mv = self.SampleVisibles01(h, β=β)

        return v, mv, h, mh

    def updateWeights(
        self,
        v_pos: torch.Tensor,
        h_pos: torch.Tensor,
        v_neg: torch.Tensor,
        h_neg_v: torch.Tensor,
        h_neg_m: torch.Tensor,
    ):
        """
        #TODO
        Update weights and biases

        Parameters:
        -----------
        v_pos : torch.tensor of shape ()
            Visible values for the positive part of the gradient

        h_pos : torch.tensor of shape ()
            Hidden values for the positive part of the gradient

        v_neg : torch.tensor of shape ()
            Visible values for the negative part of the gradient

        h_neg_v : torch.tensor of shape ()
            

        h_neg_m : torch.tensor of shape ()

        """

        lr_p = self.lr / self.mb_s
        lr_n = self.lr / self.num_pcd

        NegTerm_ia = h_neg_v.mm(v_neg.t())

        self.W += h_pos.mm(v_pos.t()) * lr_p - NegTerm_ia * lr_n
        self.vbias += torch.sum(v_pos, 1) * lr_p - torch.sum(v_neg, 1) * lr_n
        self.hbias += torch.sum(h_pos, 1) * lr_p - torch.sum(h_neg_m, 1) * lr_n

    def updateWeightsCentered(
        self,
        v_pos: torch.Tensor,
        h_pos_m: torch.Tensor,
        v_neg: torch.Tensor,
        h_neg_m: torch.Tensor,
    ):
        """
        #TODO
        Update weights and biases using Centered gradient update

        Parameters:
        -----------
        v_pos : torch.tensor of shape ()
            Visible values for the positive part of the gradient

        h_pos_m : torch.tensor of shape ()
            Hidden average values for the positive part of the gradient

        v_neg : torch.tensor of shape ()
            Visible values for the negative part of the gradient

        h_neg_m : torch.tensor of shape ()
            Hidden average values for the negative part of the gradient 
        """
        self.VisDataAv = torch.mean(v_pos, 1)
        self.HidDataAv = torch.mean(h_pos_m, 1)
        Xc_pos = (v_pos.t() - self.VisDataAv).t()
        Hc_pos = (h_pos_m.t() - self.HidDataAv).t()

        Xc_neg = (v_neg.t() - self.VisDataAv).t()
        Hc_neg = (h_neg_m.t() - self.HidDataAv).t()

        NormPos = 1.0 / self.mb_s
        NormNeg = 1.0 / self.num_pcd

        siτa_neg = Hc_neg.mm(Xc_neg.t()) * NormNeg
        si_neg = torch.sum(v_neg, 1) * NormNeg
        τa_neg = torch.sum(h_neg_m, 1) * NormNeg

        ΔW = Hc_pos.mm(Xc_pos.t()) * NormPos - siτa_neg

        self.W += ΔW * self.lr

        ΔVB = torch.sum(v_pos, 1) * NormPos - si_neg - torch.mv(ΔW.t(), self.HidDataAv)
        self.vbias += self.lr * ΔVB

        ΔHB = torch.sum(h_pos_m, 1) * NormPos - τa_neg - torch.mv(ΔW, self.VisDataAv)
        self.hbias += self.lr * ΔHB

    def fit_batch(self, X: torch.Tensor):
        """
        Fit the model to the minibatch

        Parameters:
        -----------

        X : torch.tensor of shape (Nv, mb_s)
            Minibatch
        """
        h_pos_v, h_pos_m = self.SampleHiddens01(X)
        self.mbatch = X
        if self.CDLearning:
            self.X_pc = X
            self.X_pc, _, h_neg_v, h_neg_m = self.GetAv()
        else:
            self.X_pc, _, h_neg_v, h_neg_m = self.GetAv()

        if self.UpdCentered:
            self.updateWeightsCentered(X, h_pos_v, h_pos_m, self.X_pc, h_neg_v, h_neg_m)
        else:
            self.updateWeights(X, h_pos_m, self.X_pc, h_neg_v, h_neg_m)

    def getMiniBatches(self, X: torch.Tensor, m: int):
        """
        Get minibatch in the dataset

        Parameters:
        -----------

        X : torch.tensor of shape (Nv, Ns)
            Dataset

        m : int
            Index of the minibatch to load

        Returns:
        --------

        torch.tensor of shape (Nv, mb_s)
            Minibatch
        """

        return X[:, m * self.mb_s : (m + 1) * self.mb_s]

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

        if ep_max == 0:
            ep_max = self.ep_max

        NB = int(X.shape[1] / self.mb_s)

        if self.ep_tot == 0:
            self.VisDataAv = torch.mean(X, 1)

        if (len(self.list_save_time) > 0) & (self.up_tot == 0):
            f = h5py.File(f"../model/AllParameters{self.file_stamp}.h5", "w")
            f.create_dataset(f"alltime", data=self.list_save_time)
            f.close()

        if (len(self.list_save_rbm) > 0) & (self.ep_tot == 0):
            f = h5py.File(f"../model/RBM{self.file_stamp}.h5", "w")
            f.create_dataset(f"lr", data=self.lr)
            f.create_dataset(f"NGibbs", data=self.gibbs_steps)
            f.create_dataset(f"UpdByEpoch", data=NB)
            f.create_dataset(f"miniBatchSize", data=self.mb_s)
            f.create_dataset(f"numPCD", data=self.num_pcd)
            f.create_dataset(f"alltime", data=self.list_save_rbm)
            f.close()

        for ep in range(ep_max):
            print(f"IT {self.ep_tot}")
            self.ep_tot += 1

            Xp = X[:, torch.randperm(X.size()[1])]
            for m in range(NB):
                if self.ResetPermChainBatch:
                    self.X_pc = torch.bernoulli(
                        torch.rand(
                            (self.Nv, self.num_pcd),
                            device=self.device,
                            dtype=self.dtype,
                        )
                    )

                Xb = self.getMiniBatches(Xp, m)
                self.fit_batch(Xb)

                if self.up_tot in self.list_save_time:
                    f = h5py.File(f"../model/AllParameters{self.file_stamp}.h5", "a")
                    print(f"Saving nb_upd={self.up_tot}")
                    f.create_dataset(f"W{self.up_tot}", data=self.W.cpu())
                    f.create_dataset(f"vbias{self.up_tot}", data=self.vbias.cpu())
                    f.create_dataset(f"hbias{self.up_tot}", data=self.hbias.cpu())
                    f.close()

                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                f = h5py.File("../model/RBM" + self.file_stamp + ".h5", "a")
                f.create_dataset(f"W{self.up_tot}", data=self.W.cpu())
                f.create_dataset(f"vbias{self.up_tot}", data=self.vbias.cpu())
                f.create_dataset(f"hbias{self.up_tot}", data=self.hbias.cpu())
                f.close()
