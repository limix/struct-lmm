# -*- coding: utf-8 -*-
# Adapted SKAT-O script including covariates but is generalised for cases when no covariates exist
# Does not include centering options

import scipy as sp
import scipy.linalg as la
import scipy.stats as st

from chiscore import davies_pvalue, mod_liu, optimal_davies_pvalue
from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR


def P(gp, M):
    RV = gp.covar.solve(M)
    if gp.mean.F is not None:
        WKiM = sp.dot(gp.mean.W.T, RV)
        WAiWKiM = sp.dot(gp.mean.W, gp.Areml.solve(WKiM))
        KiWAiWKiM = gp.covar.solve(WAiWKiM)
        RV -= KiWAiWKiM
    return RV


class StructLMM(object):
    r"""Mixed-model with genetic effect heterogeneity.

    The StructLMM model is

    .. math::
        \mathbf{y}=
        \underbrace{\mathbf{F}\mathbf{b}}_{\text{covariates}}+
        \underbrace{\mathbf{x}\odot\boldsymbol{\beta}}_{\text{genetics}}+
        \underbrace{\mathbf{e}}_{\text{random effect}}+
        \underbrace{\boldsymbol{\psi}}_{\text{noise}}

    where

    .. math::
        \boldsymbol{\beta}\sim\mathcal{N}(\mathbf{0},
        \sigma_g^2(
        \underbrace{\rho\boldsymbol{EE}^T}_{\text{GxE}}+
        \underbrace{(1-\rho)\mathbf{1}_{N\times N}}_{\text{persistent}}))

    .. math::
        \mathbf{e}\sim\mathcal{N}(\mathbf{0}, \sigma_e^2\mathbf{WW}^T)

    .. math::
        \boldsymbol{\psi}\sim\mathcal{N}(\mathbf{0}, \sigma_n^2\mathbf{I}_N)

    StructLMM can be used to implement

    - GxE test (persistent+GxE vs persistent, i.e. :math:`\rho\neq{0}`)
    - joint test (persistent+GxE vs no effect, i.e. :math:`\sigma_g^2>0`)

    Parameters
    ----------
    y : (`N`, 1) ndarray
        phenotype vector
    Env : (`N`, `K`)
          Environmental matrix (indviduals by number of environments)
    W : (`N`, `T`)
        design of random effect in the null model.
        By default, W is set to ``Env``.
    rho_list : list
        list of ``rho`` values.  Note that ``rho = 1-rho`` in the equation described above.
        ``rho=0`` correspond to no persistent effect (only GxE);
        ``rho=1`` corresponds to only persistent effect (no GxE);
        By default, ``rho=[0, 0.1**2, 0.2**2, 0.3**2, 0.4**2, 0.5**2, 0.5, 1.]``

    Examples
    --------
    This example shows how to run StructLMM.
    Let's start with the joint test:

    .. doctest::

        >>> from numpy.random import RandomState
        >>> import scipy as sp
        >>> from struct_lmm import StructLMM
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
        >>> rho = [0., 0.1**2, 0.2**2, 0.3**2, 0.4**2, 0.5**2, 0.5, 1.] # list of rhos
        >>>
        >>> slmm = StructLMM(y, E, W=E, rho_list=rho)
        >>> null = slmm.fit_null(F=covs, verbose=False)
        >>> pv = slmm.score_2_dof(x)
        >>> print('%.4f' % pv)
        0.4054

    and now the interaction test

        >>> hGWASint = StructLMM(y, E, rho_list=[0])
        >>> null = hGWASint.fit_null(F=sp.hstack([covs, x]), verbose=False)
        >>> pv = hGWASint.score_2_dof(x)
        >>> print('%.4f' % pv)
        0.3294
    """

    def __init__(self, y, Env, K=None, W=None, rho_list=None):
        self.y = y
        self.Env = Env
        if W is None:
            W = Env
        # K is kernel under the null (exclusing noise)
        self.K = K
        # W is low rank verion of kernel
        self.W = W
        # rho_list here is 1-rho used in documentation and paper but used in same manner as for SKAT-O
        self.rho_list = rho_list
        if self.rho_list is None:
            self.rho_list = [
                0.0,
                0.1 ** 2,
                0.2 ** 2,
                0.3 ** 2,
                0.4 ** 2,
                0.5 ** 2,
                0.5,
                1.0,
            ]
        if len(sp.where(sp.array(self.rho_list) == 1)[0]) > 0:
            self.rho_list[sp.where(sp.array(self.rho_list) == 1)[0][0]] = 0.999
        self.vec_ones = sp.ones((1, self.y.shape[0]))

    def fit_null(self, F=None, verbose=True):
        """
        Parameters
        ----------
        F : (`N`, L) ndarray
            fixed effect design for covariates.

        Returns
        -------
        RV : dict
             Dictionary with null model info (TODO add details)
        """
        #  F is a fixed effect covariate matrix with dim = N by D
        #  F itself cannot have any cols of 0's and it won't work if it is None
        self.F = F
        if self.K is not None:
            # Decompose K into low rank version
            S_K, U_K = la.eigh(self.K)
            S = sp.array([i for i in S_K if i > 1e-9])
            U = U_K[:, -len(S) :]
            # In most cases W = E but have left it as seperate parameter for
            # flexibility
            self.W = U * S ** 0.5
            self.gp = GP2KronSumLR(
                Y=self.y, F=self.F, A=sp.eye(1), Cn=FreeFormCov(1), G=self.W
            )
            self.gp.covar.Cr.setCovariance(0.5 * sp.ones((1, 1)))
            self.gp.covar.Cn.setCovariance(0.5 * sp.ones((1, 1)))
            RV = self.gp.optimize(verbose=verbose)
            #  Get optimal kernel parameters
            self.covarparam0 = self.gp.covar.Cr.K()[0, 0]
            self.covarparam1 = self.gp.covar.Cn.K()[0, 0]
            self.Kiy = self.gp.Kiy()
        elif self.W is not None:
            self.gp = GP2KronSumLR(
                Y=self.y, F=self.F, A=sp.eye(1), Cn=FreeFormCov(1), G=self.W
            )
            self.gp.covar.Cr.setCovariance(0.5 * sp.ones((1, 1)))
            self.gp.covar.Cn.setCovariance(0.5 * sp.ones((1, 1)))
            RV = self.gp.optimize(verbose=verbose)
            self.covarparam0 = self.gp.covar.Cr.K()[0, 0]  # getParams()[0]
            self.covarparam1 = self.gp.covar.Cn.K()[0, 0]
            self.Kiy = self.gp.Kiy()
        else:
            # If there is no kernel then solve analytically
            self.alpha_hat = sp.dot(
                sp.dot(la.inv(sp.dot(self.F.T, self.F)), self.F.T), self.y
            )
            yminus_falpha_hat = self.y - sp.dot(self.F, self.alpha_hat)
            self.covarparam1 = (yminus_falpha_hat ** 2).sum() / yminus_falpha_hat.shape[
                0
            ]
            self.covarparam0 = 0
            self.Kiy = (1 / float(self.covarparam1)) * self.y
            self.W = sp.zeros(self.y.shape)
            RV = self.covarparam0
        return RV

    def score_2_dof(self, X, snp_dim="col", debug=False):
        """
        Parameters
        ----------
        X : (`N`, `1`) ndarray
            genotype vector (TODO: X should be small)

        Returns
        -------
        pvalue : float
            P value
        """
        # 1. calculate Qs and pvs
        Q_rho = sp.zeros(len(self.rho_list))
        Py = P(self.gp, self.y)
        for i in range(len(self.rho_list)):
            rho = self.rho_list[i]
            LT = sp.vstack((rho ** 0.5 * self.vec_ones, (1 - rho) ** 0.5 * self.Env.T))
            LTxoPy = sp.dot(LT, X * Py)
            Q_rho[i] = 0.5 * sp.dot(LTxoPy.T, LTxoPy)

        # Calculating pvs is split into 2 steps
        # If we only consider one value of rho i.e. equivalent to SKAT and used for interaction test
        if len(self.rho_list) == 1:
            rho = self.rho_list[0]
            L = sp.hstack((rho ** 0.5 * self.vec_ones.T, (1 - rho) ** 0.5 * self.Env))
            xoL = X * L
            PxoL = P(self.gp, xoL)
            LToxPxoL = 0.5 * sp.dot(xoL.T, PxoL)
            pval = davies_pvalue(Q_rho[0], LToxPxoL)
            # Script ends here for interaction test
            return pval
        # or if we consider multiple values of rho i.e. equivalent to SKAT-O and used for association test
        else:
            pliumod = sp.zeros((len(self.rho_list), 4))
            for i in range(len(self.rho_list)):
                rho = self.rho_list[i]
                L = sp.hstack(
                    (rho ** 0.5 * self.vec_ones.T, (1 - rho) ** 0.5 * self.Env)
                )
                xoL = X * L
                PxoL = P(self.gp, xoL)
                LToxPxoL = 0.5 * sp.dot(xoL.T, PxoL)
                eighQ, UQ = la.eigh(LToxPxoL)
                pliumod[i,] = mod_liu(Q_rho[i], eighQ)
            T = pliumod[:, 0].min()
            # if optimal_rho == 0.999:
            #    optimal_rho = 1

            # 2. Calculate qmin
            qmin = sp.zeros(len(self.rho_list))
            percentile = 1 - T
            for i in range(len(self.rho_list)):
                q = st.chi2.ppf(percentile, pliumod[i, 3])
                # Recalculate p-value for each Q rho of seeing values at least as extreme as q again using the modified matching moments method
                qmin[i] = (q - pliumod[i, 3]) / (2 * pliumod[i, 3]) ** 0.5 * pliumod[
                    i, 2
                ] + pliumod[i, 1]

            # 3. Calculate quantites that occur in null distribution
            Px1 = P(self.gp, X)
            m = 0.5 * sp.dot(X.T, Px1)
            xoE = X * self.Env
            PxoE = P(self.gp, xoE)
            ETxPxE = 0.5 * sp.dot(xoE.T, PxoE)
            ETxPx1 = sp.dot(xoE.T, Px1)
            ETxPx11xPxE = 0.25 / m * sp.dot(ETxPx1, ETxPx1.T)
            ZTIminusMZ = ETxPxE - ETxPx11xPxE
            eigh, vecs = la.eigh(ZTIminusMZ)

            eta = sp.dot(ETxPx11xPxE, ZTIminusMZ)
            vareta = 4 * sp.trace(eta)

            OneZTZE = 0.5 * sp.dot(X.T, PxoE)
            tau_top = sp.dot(OneZTZE, OneZTZE.T)
            tau_rho = sp.zeros(len(self.rho_list))
            for i in range(len(self.rho_list)):
                tau_rho[i] = self.rho_list[i] * m + (1 - self.rho_list[i]) / m * tau_top

            MuQ = sp.sum(eigh)
            VarQ = sp.sum(eigh ** 2) * 2 + vareta
            KerQ = sp.sum(eigh ** 4) / (sp.sum(eigh ** 2) ** 2) * 12
            Df = 12 / KerQ

            # 4. Integration
            pvalue = optimal_davies_pvalue(
                qmin, MuQ, VarQ, KerQ, eigh, vareta, Df, tau_rho, self.rho_list, T
            )

            # Final correction to make sure that the p-value returned is sensible
            multi = 3
            if len(self.rho_list) < 3:
                multi = 2
            idx = sp.where(pliumod[:, 0] > 0)[0]
            pval = pliumod[:, 0].min() * multi
            if pvalue <= 0 or len(idx) < len(self.rho_list):
                pvalue = pval
            if pvalue == 0:
                if len(idx) > 0:
                    pvalue = pliumod[:, 0][idx].min()

            if debug:
                info = {
                    "Qs": Q_rho,
                    "pvs_liu": pliumod,
                    "qmin": qmin,
                    "MuQ": MuQ,
                    "VarQ": VarQ,
                    "KerQ": KerQ,
                    "lambd": eigh,
                    "VarXi": vareta,
                    "Df": Df,
                    "tau": tau_rho,
                }
                return pvalue, info
            else:
                return pvalue
