# -*- coding: utf-8 -*-
import pdb

import scipy as sp
import scipy.linalg as la
import scipy.stats as st
from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR


class BF():
    r"""
    Calculates BF between full model (including all environments used for the analysis) and models with environments removed

    Parameters
    ----------
    y : (`N`, 1) ndarray
        phenotype vector
    x : (`N`, 1)
        SNP vector
    F : (`N`, L) ndarray
        fixed effect design for covariates.
    Env : (`N`, `K`)
        Environmental matrix (indviduals by number of environments)
    W : (`N`, `T`)
        design of random effect in the null model.
        By default, W is set to ``Env``.

    Examples
    --------
    This example shows how to run BF.
    .. doctest::

        >>> from numpy.random import RandomState
        >>> import scipy as sp
        >>> from struct_lmm.interpretation import BF
        >>> random = RandomState(1)
        >>>
        >>> # generate data
        >>> n = 20 # number samples
        >>> k = 4 # number environments
        >>>
        >>> y = random.randn(n, 1) # phenotype
        >>> x = 1. * (random.rand(n, 1) < 0.2) # genotype
        >>> E = random.randn(n, k) # environemnts
        >>> covs = sp.ones((n, 1)) # intercept
        >>>
        >>> bf = BF(y, x, F = covs, Env = E, W=E)
        >>> persistent_model = bf.calc_persistent_model()
        >>> full_model = bf.calc_full_model()
        >>> marginal_model = bf.calc_marginal_model()
        >>> print('%.4f' % persistent_model, '%.4f' % full_model, '%.4f' % marginal_model)
        ('-16.1504', '-16.1504', '-16.1504')
    """

    def __init__(self, y, x, F, Env, W=None):
        self.y = y
        self.x = x
        self.F = F
        self.Env = Env
        self.W = W
        if self.W is None: self.W = self.Env

    def calc_persistent_model(self):
        _covs = sp.concatenate([self.F, self.W, self.x], 1)
        xoE = sp.ones(self.x.shape)
        gp=GP2KronSumLR(Y=self.y, F=_covs, A=sp.eye(1), Cn=FreeFormCov(1), G=xoE)
        gp.covar.Cr.setCovariance(1e-4 * sp.ones((1,1)))
        gp.covar.Cn.setCovariance(0.02 * sp.ones((1,1)))
        RV = gp.optimize(verbose=False)
        lml = -gp.LML()

        return lml

    def calc_full_model(self):
        _covs = sp.concatenate([self.F, self.W, self.x], 1)
        xoE = self.x * self.Env
        gp=GP2KronSumLR(Y=self.y, F=_covs, A=sp.eye(1), Cn=FreeFormCov(1), G=xoE)
        gp.covar.Cr.setCovariance(1e-4 * sp.ones((1,1)))
        gp.covar.Cn.setCovariance(0.02 * sp.ones((1,1)))
        RV = gp.optimize(verbose=False)
        lml = -gp.LML()

        return lml    

    def calc_marginal_model(self, env_remove = 0):
        _covs = sp.concatenate([self.F, self.W, self.x], 1)
        Env_subset = sp.delete(self.Env, env_remove, axis = 1)
        xoE = self.x * Env_subset
        gp=GP2KronSumLR(Y=self.y, F=_covs, A=sp.eye(1), Cn=FreeFormCov(1), G=xoE)
        gp.covar.Cr.setCovariance(1e-4 * sp.ones((1,1)))
        gp.covar.Cn.setCovariance(0.02 * sp.ones((1,1)))
        RV = gp.optimize(verbose=False)
        lml = -gp.LML()

        return lml    
