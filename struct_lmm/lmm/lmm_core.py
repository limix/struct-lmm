import pdb
import time

import scipy as sp
import scipy.linalg as la
import scipy.stats as st

import limix


def calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
    Ai = la.pinv(FKiF)
    beta = sp.dot(Ai, FKiy)
    s2 = (yKiy - sp.dot(FKiy[:, 0], beta[:, 0])) / df
    return Ai, beta, s2


def hatodot(A, B):
    """ should be implemented in C """
    A1 = sp.kron(A, sp.ones((1, B.shape[1])))
    B1 = sp.kron(sp.ones((1, A.shape[1])), B)
    return A1 * B1


class LMMCore():
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

        .. doctest::

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
            >>> y = random.randn(N,1)
            >>> E = random.randn(N,k)
            >>> G = 1.*(random.rand(N,S)<0.2)
            >>> F = sp.concatenate([sp.ones((N,1)), random.randn(N,1)], 1)
            >>> Inter = random.randn(N, m)
            >>>
            >>> gp = GP2KronSumLR(Y=y, Cn=FreeFormCov(1), G=E, F=F, A=sp.ones((1,1)))
            >>> gp.covar.Cr.setCovariance(0.5*sp.ones((1,1)))
            >>> gp.covar.Cn.setCovariance(0.5*sp.ones((1,1)))
            >>> info_null = gp.optimize(verbose=False)
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
            [0.1428 0.3932 0.4749 0.3121]
            >>> print(beta[:,:4])
            [[ 0.0535  0.2571  0.122  -0.2328]
             [ 0.3418  0.3179 -0.2099  0.0986]]

        Example with step=4

        .. doctest::

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
            [0.636  0.3735 0.7569 0.3282]
            >>> print(beta[:,:4])
            [[-0.0176 -0.4057  0.0925 -0.4175]
             [ 0.2821  0.2232 -0.2124  0.2433]
             [-0.0575  0.1528 -0.0384  0.019 ]
             [-0.2156 -0.2327 -0.1773 -0.1497]]

        Example with step=4 and Inter

        .. doctest::

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
            [0.1591 0.2488 0.4109 0.5877]
            >>> print(beta[:,:4])
            [[ 0.0691  0.1977  0.435  -0.3214]
             [ 0.1701 -0.3195  0.0179  0.3344]
             [ 0.1887 -0.0765 -0.3847  0.1843]
             [-0.2974  0.2787  0.2427 -0.0717]
             [ 0.3784  0.0854  0.1566  0.0652]
             [ 0.4012  0.5255 -0.1572  0.1674]
             [-0.413   0.0278  0.1946 -0.1199]
             [ 0.0268 -0.0317 -0.1059  0.1414]]
    """

    def __init__(self, y, F, cov=None):
        if F is None: F = sp.ones((y.shape[0], 1))
        self.y = y
        self.F = F
        self.cov = cov
        self.df = y.shape[0] - F.shape[1]
        self._fit_null()

    def _fit_null(self):
        """ Internal functon. Fits the null model """
        if self.cov == None:
            self.Kiy = self.y
            self.KiF = self.F
        else:
            self.Kiy = self.cov.solve(self.y)
            self.KiF = self.cov.solve(self.F)
        self.FKiy = sp.dot(self.F.T, self.Kiy)
        self.FKiF = sp.dot(self.F.T, self.KiF)
        self.yKiy = sp.dot(self.y[:, 0], self.Kiy[:, 0])
        # calc beta_F0 and s20
        self.A0i, self.beta_F0, self.s20 = calc_Ai_beta_s2(
            self.yKiy, self.FKiF, self.FKiy, self.df)

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
        else: mi = Inter.shape[1]
        k = self.F.shape[1]
        m = mi * step
        F1KiF1 = sp.zeros((k + m, k + m))
        F1KiF1[:k, :k] = self.FKiF
        F1Kiy = sp.zeros((k + m, 1))
        F1Kiy[:k, 0] = self.FKiy[:, 0]
        s2 = sp.zeros(ntests)
        self.beta_g = sp.zeros([m, ntests])
        for s in range(ntests):
            idx1 = step * s
            idx2 = step * (s + 1)
            if Inter is not None:
                if step == 1:
                    X = Inter * G[:, idx1:idx2]
                else:
                    X = hatodot(Inter, G[:, idx1:idx2])
            else:
                X = G[:, idx1:idx2]
            if self.cov == None: KiX = X
            else: KiX = self.cov.solve(X)
            F1KiF1[k:, :k] = sp.dot(X.T, self.KiF)
            F1KiF1[:k, k:] = F1KiF1[k:, :k].T
            F1KiF1[k:, k:] = sp.dot(X.T, KiX)
            F1Kiy[k:, 0] = sp.dot(X.T, self.Kiy[:, 0])
            #this can be sped up by using block matrix inversion, etc
            _, beta, s2[s] = calc_Ai_beta_s2(self.yKiy, F1KiF1, F1Kiy, self.df)
            self.beta_g[:, s] = beta[k:, 0]
        #dlml and pvs
        self.lrt = -self.df * sp.log(s2 / self.s20)
        self.pv = st.chi2(m).sf(self.lrt)

        t1 = time.time()
        if verbose:
            print('Tested for %d variants in %.2f s' % (G.shape[1], t1 - t0))

    def getPv(self):
        """
        Get pvalues

        Returns
        -------
        pv : ndarray
        """
        return self.pv

    def getBetaSNP(self):
        """
        get effect size SNPs


        Returns
        -------
        pv : ndarray
        """
        return self.beta_g

    def getBetaCov(self):
        """
        get beta of covariates

        Returns
        -------
        beta : ndarray
        """
        return self.beta_F

    def getLRT(self):
        """
        get lik ratio test statistics

        Returns
        -------
        lrt : ndarray
        """
        return self.lrt

    def getBetaSNPste(self):
        """
        get standard errors on betas

        Returns
        -------
        beta_ste : ndarray
        """
        beta = self.getBetaSNP()
        pv = self.getPv()
        z = sp.sign(beta) * sp.sqrt(st.chi2(1).isf(pv))
        ste = beta / z
        return ste
