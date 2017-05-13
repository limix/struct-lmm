import scipy as sp
import scipy.stats as st
import scipy.linalg as la
import pdb
import limix
import time
from struct_lmm.lmm import LMM

def calc_Ai_beta_s2(yKiy,FKiF,FKiy,df):
    Ai = la.pinv(FKiF)
    beta = sp.dot(Ai,FKiy)
    s2 = (yKiy-sp.dot(FKiy[:,0],beta[:,0]))/df
    return Ai,beta,s2

class LMMCore(LMM):
    r"""
    Core LMM for interaction and association testing.

    The model is A CAPOCCHIA

    .. math::
        \mathbf{y}\sim\mathcal{N}(
        \underbrace{\mathbf{F}\mathbf{b}}_{\text{covariates}}+
        \underbrace{(\mathbf{X}\hat{\odot}\mathbf{I})\mathbf{
        \beta}}_{\text{genetics}},
        \underbrace{\mathbf{K}_{\boldsymbol{\theta}}}_{
        \text{covariance}})

    Where :math:`\hat{\odot}` is defined as follows.
    Given
    :math:`\mathbf{A}=[\mathbf{a}_1, \dots, \mathbf{a}_{m}]\in\mathbb{R}^{n\times{m}}`  and
    :math:`\mathbf{B}=[\mathbf{b}_1, \dots, \mathbf{b}_{l}]\in\mathbb{R}^{n\times{l}}`, then
    :math:`\mathbf{A}\hat{\odot}\mathbf{B}=[\mathbf{a}_1\odot\mathbf{b}_1,\dots,\mathbf{a}_1\odot\mathbf{b}_{l},\mathbf{a}_2\odot\mathbf{b}_1,\dots,\mathbf{a}_2\odot\mathbf{b}_{l},\dots,\mathbf{a}_{m}\odot\mathbf{b}_{l},\dots,\mathbf{a}_{m}\odot\mathbf{b}_{l}]`

    The test :math:`\mathbf{\beta}\neq{0}` is done for all
    provided variants one-by-one.

    Parameters
    ----------
    y : (`N`, 1) ndarray
        phenotype vector
    F : (`N`, L) ndarray
        fixed effect design for covariates.
    cov : :class:`limix_core.covar`
        Covariance matrix of the random effect

    Examples
    --------

        >>> from numpy.random import RandomState
        >>> import scipy as sp
        >>> from struct_lmm import LMMInterCore
        >>> from limix_core.gp import GP2KronSumLR
        >>> from limix_core.covar import FreeFormCov
        >>> random = RandomState(1)
        >>> from numpy import set_printoptions
        >>> set_printoptions(4)
        >>>
        >>> N = 100
        >>> k = 1
        >>> m = 2
        >>> S = 1000
        >>> y = sp.randn(N,1)
        >>> E = sp.randn(N,k)
        >>> G = 1.*(sp.rand(N,S)<0.2)
        >>> F = sp.concatenate([sp.ones((N,1)), sp.randn(N,1)], 1)
        >>> Inter = sp.randn(N, m)
        >>>
        >>> gp = GP2KronSumLR(Y=y, Cn=FreeFormCov(1), G=E, F=F, A=sp.ones((1,1)))
        >>> gp.covar.Cr.setCovariance(0.5*sp.ones((1,1)))
        >>> gp.covar.Cn.setCovariance(0.5*sp.ones((1,1)))
        >>> gp.optimize()
        >>>
        >>> lmm = LMMInterCore(y, F, gp.covar)
        >>> lmm.process(G, Inter)
        >>> pv = lmm.getPv()
        >>> beta = lmm.getBetaSNP()
        >>> lrt = lmm.getLRT()
    """

    def _fit_null(self):
        """ Internal functon. Fits the null model """
        if self.cov==None:
            self.Kiy = self.y
            self.KiF = self.F
        else:
            self.Kiy = self.cov.solve(self.y)
            self.KiF = self.cov.solve(self.F)
        self.FKiy = sp.dot(self.F.T, self.Kiy)
        self.FKiF = sp.dot(self.F.T, self.KiF)
        self.yKiy = sp.dot(self.y[:,0], self.Kiy[:,0])
        # calc beta_F0 and s20
        self.A0i, self.beta_F0, self.s20 = calc_Ai_beta_s2(self.yKiy,self.FKiF,self.FKiy,self.df)

    def process(self, G, Inter, verbose=False):
        r"""
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
        Inter : (`N`, `M`) ndarray
            Matrix of `M` factors for `N` inds with which 
            each variant interact
        verbose : bool
            verbose flag.
        """
        t0 = time.time()
        k = self.F.shape[1]
        m = Inter.shape[1]
        F1KiF1 = sp.zeros((k+m, k+m))
        F1KiF1[:k,:k] = self.FKiF
        F1Kiy = sp.zeros((k+m,1))
        F1Kiy[:k,0] = self.FKiy[:,0] 
        s2 = sp.zeros(G.shape[1])
        self.beta_g = sp.zeros([m,G.shape[1]])
        for s in range(G.shape[1]):
            X = G[:,[s]]*Inter
            if self.cov==None:  KiX = X 
            else:               KiX = self.cov.solve(X)
            F1KiF1[k:,:k] = sp.dot(X.T,self.KiF)
            F1KiF1[:k,k:] = F1KiF1[k:,:k].T
            F1KiF1[k:,k:] = sp.dot(X.T, KiX) 
            F1Kiy[k:,0] = sp.dot(X.T,self.Kiy[:,0])
            #this can be sped up by using block matrix inversion, etc
            _,beta,s2[s] = calc_Ai_beta_s2(self.yKiy,F1KiF1,F1Kiy,self.df)
            self.beta_g[:,s] = beta[k:,0]
        #dlml and pvs
        self.lrt = -self.df*sp.log(s2/self.s20)
        self.pv = st.chi2(m).sf(self.lrt)

        t1 = time.time()
        if verbose:
            print 'Tested for %d variants in %.2f s' % (G.shape[1],t1-t0)
