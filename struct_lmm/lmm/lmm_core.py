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

def hatodot(A, B):
    """ should be implemented in C """
    A1 = sp.kron(A, sp.ones((1, B.shape[1])))
    B1 = sp.kron(sp.ones((1, A.shape[1])), B)
    return A1*B1

class LMMCore(LMM):
    r"""
    Core LMM for interaction and association testing.

    The model is

    .. math::
        \mathbf{y}\sim\mathcal{N}(
        \underbrace{\mathbf{F}\mathbf{b}}_{\text{covariates}}+
        \underbrace{(\mathbf{X}\hat{\odot}\mathbf{I})\;
        \boldsymbol{\beta}}_{\text{genetics}},
        \sigma^2\underbrace{\mathbf{K}_{
        \boldsymbol{\theta}_0}}_{\text{covariance}})

    :math:`\hat{\odot}` is defined as follows.
    Let

    .. math::
        \mathbf{A}=\left[\mathbf{a}_1,
                         \dots,
                         \mathbf{a}_{m}
                         \right]\in\mathbb{R}^{n\times{m}}]

    and

    .. math::
        \mathbf{B}=\left[\mathbf{b}_1,
                         \dots,
                         \mathbf{b}_{l}
                         \right]\in\mathbb{R}^{n\times{l}}

    then

    .. math::
        \mathbf{A}\hat{\odot}\mathbf{B} =
         \left[\mathbf{a}_1\odot\mathbf{b}_1
               \dots,
               \mathbf{a}_1\odot\mathbf{b}_{l},
               \mathbf{a}_2\odot\mathbf{b}_1,
               \dots,
               \mathbf{a}_2\odot\mathbf{b}_{l},
               \dots,
               \mathbf{a}_{m}\odot\mathbf{b}_{l},
               \dots,\mathbf{a}_{m}\odot\mathbf{b}_{l}
               \right]\in\mathbb{R}^{n\times{(ml)}}

    where :math:`\odot` is the element-wise product.

    :math:`\mathbf{K}_{\boldsymbol{\theta}_0}` is specified
    by a :class:`limix_core.covar`.
    Within the function only :math:`\sigma^2` is learnt.
    The covariance parameters :math:`\boldsymbol{\theta}_0`
    should then be set (or learnt) externally.

    The test :math:`\boldsymbol{\beta}\neq{0}` is done
    in bocks of ``step`` variants,
    where ``step`` can be specifed by the user.

    Parameters
    ----------
    y : (`N`, 1) ndarray
        phenotype vector
    F : (`N`, L) ndarray
        fixed effect design for covariates.
    cov : :class:`limix_core.covar`
        Covariance matrix of the random effect.
        

    Examples
    --------

        Example with Inter and step=1

        >>> from numpy.random import RandomState
        >>> import scipy as sp
        >>> from struct_lmm import LMMCore
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
        >>> lmm = LMMCore(y, F, gp.covar)
        >>> lmm.process(G, Inter)
        >>> pv = lmm.getPv()
        >>> beta = lmm.getBetaSNP()
        >>> beta_ste = lmm.getBetaSNPste()
        >>> lrt = lmm.getLRT()
        >>>
        >>> print(pv.shape)
        (1000,)
        >>> print(beta.shape)
        (2, 1000)
        >>> print(pv[:4])
        [ 0.4357  0.9631  0.6682  0.1345]
        >>> print(beta[:,:4])
        [[ 0.2393  0.011  -0.1154 -0.2802]
         [ 0.0606  0.0731 -0.1291 -0.2974]]

        Example with step=4

        >>> lmm.process(G, step=4)
        >>> pv = lmm.getPv()
        >>> beta = lmm.getBetaSNP()
        >>> lrt = lmm.getLRT()
        >>>
        >>> print(pv.shape)
        (250,)
        >>> print(beta.shape)
        (4, 250)
        >>> print(pv[:4])
        [ 0.466   0.509   0.2587  0.0242]
        >>> print(beta[:,:4])
        [[-0.0827 -0.2116 -0.4932  0.1934]
         [-0.3204 -0.0287 -0.1827 -0.2561]
         [ 0.2059  0.0777 -0.0613 -0.582 ]
         [ 0.1783 -0.4189 -0.2429 -0.1931]]

        Example with step=4 and Inter

        >>> lmm = LMMCore(y, F, gp.covar)
        >>> lmm.process(G, Inter=Inter, step=4)
        >>> pv = lmm.getPv()
        >>> beta = lmm.getBetaSNP()
        >>> lrt = lmm.getLRT()
        >>>
        >>> print(pv.shape)
        (250,)
        >>> print(beta.shape)
        (8, 250)
        >>> print(pv[:4])
        [ 0.5771  0.1618  0.433   0.6666]
        >>> print(beta[:,:4])
        [[ 0.2866  0.2929 -0.1993 -0.0407]
         [ 0.0339 -0.3662 -0.358  -0.0709]
         [-0.0095 -0.1869 -0.086  -0.1147]
         [-0.3332 -0.243  -0.2058  0.0868]
         [ 0.0879  0.2488  0.169   0.0294]
         [ 0.0903 -0.0867  0.0657 -0.4517]
         [-0.0621  0.0466  0.0709  0.1487]
         [-0.2597 -0.7184 -0.272  -0.1507]]
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

    def process(self, G, Inter=None, step=1, verbose=False):
        r"""
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
        Inter : (`N`, `M`) ndarray
            Matrix of `M` factors for `N` inds with which 
            each variant interact
            By default, Inter is set to a matrix of ones.
        step : int
            Number of consecutive variants that should be
            tested jointly.
        verbose : bool
            verbose flag.
        """
        t0 = time.time()
        ntests = int(G.shape[1] / step)
        if Inter is None: mi = 1
        else:             mi = Inter.shape[1]
        k = self.F.shape[1]
        m = mi * step 
        F1KiF1 = sp.zeros((k+m, k+m))
        F1KiF1[:k,:k] = self.FKiF
        F1Kiy = sp.zeros((k+m,1))
        F1Kiy[:k,0] = self.FKiy[:,0] 
        s2 = sp.zeros(ntests)
        self.beta_g = sp.zeros([m, ntests])
        for s in range(ntests):
            idx1 = step * s
            idx2 = step * (s + 1) 
            if Inter is not None:
                if step==1:
                    X = Inter * G[:,idx1:idx2]
                else:
                    X = hatodot(Inter, G[:,idx1:idx2])
            else:
                X = G[:,idx1:idx2]
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
