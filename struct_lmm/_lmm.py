# Adapted SKAT-O script including covariates but is generalised for cases when no covariates exist
# Does not include centering options


from chiscore import davies_pvalue, mod_liu, optimal_davies_pvalue


def P(gp, M):
    from numpy_sugar.linalg import rsolve
    from scipy.linalg import cho_solve

    RV = rsolve(gp.covariance(), M)
    if gp.X is not None:
        WKiM = gp.M.T @ RV
        terms = gp._terms
        WAiWKiM = gp.X @ cho_solve(terms["Lh"], WKiM)
        KiWAiWKiM = rsolve(gp.covariance(), WAiWKiM)
        RV -= KiWAiWKiM
    return RV


class StructLMM2:
    r"""
    Structured linear mixed model that accounts for genotype-environment interactions.

    StructLMM [MC18]_ extends the conventional linear mixed model by including an
    additional per-individual effect term that accounts for genotype-environment
    interaction, which can be represented as an n×1 vector, 𝛃.
    The model can be cast as

        𝐲 = 𝙼𝛂 + 𝐠⊙𝛃 + 𝐞 + 𝛆,

    where

        𝛃∼𝓝(𝟎, 𝓋₀(ρ𝟏 + (1-ρ)𝙴𝙴ᵀ)), 𝐞∼𝓝(𝟎, 𝓋₁𝚆𝚆ᵀ), and 𝛆∼𝓝(𝟎, 𝓋₂𝙸).

    The parameters of the model are ρ, 𝓋₀, 𝓋₁, and 𝓋₂.

    References
    ----------
    .. [MC18] Moore, R., Casale, F. P., Bonder, M. J., Horta, D., Franke, L., Barroso, I., & Stegle, O. (2018). A linear mixed-model approach to study multivariate gene–environment interactions (p. 1). Nature Publishing Group.
    """

    def __init__(self, y, M, E, W=None):
        from numpy import sqrt, asarray, atleast_2d
        from numpy_sugar import ddot

        self._y = atleast_2d(asarray(y, float).ravel()).T
        self._E = atleast_2d(asarray(E, float).T).T

        if W is None:
            self._W = self._E
        elif isinstance(W, tuple):
            # W must be an eigen-decomposition of 𝚆𝚆ᵀ
            self._W = ddot(W[0], sqrt(W[1]))
        else:
            self._W = atleast_2d(asarray(W, float).T).T

        self._M = atleast_2d(asarray(M, float).T).T

        nsamples = len(self._y)
        if nsamples != self._M.shape[0]:
            raise ValueError("Number of samples mismatch between y and M.")

        if nsamples != self._E.shape[0]:
            raise ValueError("Number of samples mismatch between y and E.")

        if nsamples != self._W.shape[0]:
            raise ValueError("Number of samples mismatch between y and W.")

        self._lmm = None

        self._rhos = [0.0, 0.1 ** 2, 0.2 ** 2, 0.3 ** 2, 0.4 ** 2, 0.5 ** 2, 0.5, 1.0]

    def fit(self, verbose=True):
        from glimix_core.lmm import Kron2Sum

        self._lmm = Kron2Sum(self._y, [[1]], self._M, self._W, restricted=True)
        self._lmm.fit(verbose=verbose)
        self._covarparam0 = self._lmm.C0[0, 0]
        self._covarparam1 = self._lmm.C1[0, 0]

    def _xBy(self, rho, y, x):
        """
        Let 𝙱 = ρ𝟏 + (1-ρ)𝙴𝙴ᵀ.
        It computes 𝐲ᵀ𝙱𝐱.
        """
        l = rho * (y.sum() * x.sum())
        r = (1 - rho) * (y.T @ self._E) @ (self._E.T @ x)
        return l + r

    def _Q_rho(self, X):
        from numpy import zeros

        # 1. calculate Qs and pvs
        # Py*X (ρ𝙴𝙴ᵀ + (1-ρ)𝟏) X*Py / 2
        Q_rho = zeros(len(self._rhos))
        Py = self._P(self._y)
        XPy = X * Py
        for i in range(len(self._rhos)):
            rho = self._rhos[i]
            Q_rho[i] = self._xBy(rho, XPy, XPy) / 2

        return Q_rho

    def _pliumod(self, X, Q_rho):
        import scipy.linalg as la
        from numpy import zeros, block, ones
        from math import sqrt
        from numpy_sugar import ddot

        vec_ones = ones((1, self._y.shape[0]))

        pliumod = zeros((len(self._rhos), 4))
        for i in range(len(self._rhos)):
            rho = self._rhos[i]

            ones = sqrt(rho) * vec_ones.T
            E = sqrt(1 - rho) * self._E
            dX = X.ravel()
            RR = [
                [
                    ddot(ones.T, dX) @ self._P(ddot(ones.T, dX).T),
                    ddot(ones.T, dX) @ self._P(ddot(E.T, dX).T),
                ],
                [
                    ddot(E.T, dX) @ self._P(ddot(ones.T, dX).T),
                    ddot(E.T, dX) @ self._P(ddot(E.T, dX).T),
                ],
            ]
            LToxPxoL = block(RR) / 2

            eighQ, _ = la.eigh(LToxPxoL)
            pliumod[i, :] = mod_liu(Q_rho[i], eighQ)

        return pliumod

    def _qmin(self, pliumod):
        from numpy import zeros
        import scipy.stats as st

        # 2. Calculate qmin
        qmin = zeros(len(self._rhos))
        percentile = 1 - pliumod[:, 0].min()
        for i in range(len(self._rhos)):
            q = st.chi2.ppf(percentile, pliumod[i, 3])
            # Recalculate p-value for each Q rho of seeing values at least as
            # extreme as q again using the modified matching moments method
            qmin[i] = (q - pliumod[i, 3]) / (2 * pliumod[i, 3]) ** 0.5 * pliumod[
                i, 2
            ] + pliumod[i, 1]

        return qmin

    def _P(self, M):
        from numpy_sugar.linalg import rsolve
        from scipy.linalg import cho_solve

        RV = rsolve(self._lmm.covariance(), M)
        if self._lmm.X is not None:
            WKiM = self._lmm.M.T @ RV
            terms = self._lmm._terms
            WAiWKiM = self._lmm.X @ cho_solve(terms["Lh"], WKiM)
            KiWAiWKiM = rsolve(self._lmm.covariance(), WAiWKiM)
            RV -= KiWAiWKiM

        return RV

    def score_2_dof(self, X):
        from numpy import trace, sum
        import scipy as sp
        import scipy.linalg as la

        # 1. calculate Qs and pvs
        Q_rho = self._Q_rho(X)

        # Calculating pvs is split into 2 steps
        # If we only consider one value of rho i.e. equivalent to SKAT and used for
        # interaction test
        if len(self._rhos) == 1:
            raise NotImplementedError("We have not tested it yet.")
        # or if we consider multiple values of rho i.e. equivalent to SKAT-O and used
        # for association test
        pliumod = self._pliumod(X, Q_rho)
        qmin = self._qmin(pliumod)

        # 3. Calculate quantites that occur in null distribution
        Px1 = self._P(X)
        m = 0.5 * (X.T @ Px1)
        xoE = X * self._E
        PxoE = self._P(xoE)
        ETxPxE = 0.5 * (xoE.T @ PxoE)
        ETxPx1 = xoE.T @ Px1
        ETxPx11xPxE = 0.25 / m * (ETxPx1 @ ETxPx1.T)
        ZTIminusMZ = ETxPxE - ETxPx11xPxE
        eigh, _ = la.eigh(ZTIminusMZ)

        eta = ETxPx11xPxE @ ZTIminusMZ
        vareta = 4 * trace(eta)

        OneZTZE = 0.5 * (X.T @ PxoE)
        tau_top = OneZTZE @ OneZTZE.T
        tau_rho = sp.zeros(len(self._rhos))
        for i in range(len(self._rhos)):
            tau_rho[i] = self._rhos[i] * m + (1 - self._rhos[i]) / m * tau_top

        MuQ = sum(eigh)
        VarQ = sum(eigh ** 2) * 2 + vareta
        KerQ = sum(eigh ** 4) / (sum(eigh ** 2) ** 2) * 12
        Df = 12 / KerQ

        # 4. Integration
        T = pliumod[:, 0].min()
        pvalue = optimal_davies_pvalue(
            qmin, MuQ, VarQ, KerQ, eigh, vareta, Df, tau_rho, self._rhos, T
        )

        # Final correction to make sure that the p-value returned is sensible
        multi = 3
        if len(self._rhos) < 3:
            multi = 2
        idx = sp.where(pliumod[:, 0] > 0)[0]
        pval = pliumod[:, 0].min() * multi
        if pvalue <= 0 or len(idx) < len(self._rhos):
            pvalue = pval
        if pvalue == 0:
            if len(idx) > 0:
                pvalue = pliumod[:, 0][idx].min()

        return pvalue


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
        import scipy as sp

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
        # if len(sp.where(sp.array(self.rho_list) == 1)[0]) > 0:
        #     self.rho_list[sp.where(sp.array(self.rho_list) == 1)[0][0]] = 0.999
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
        import scipy as sp
        import scipy.linalg as la
        from glimix_core.lmm import Kron2Sum

        #  F is a fixed effect covariate matrix with dim = N by D
        #  F itself cannot have any col of 0's and it won't work if it is None
        self.F = F
        if self.K is not None:
            # Decompose K into low rank version
            S_K, U_K = la.eigh(self.K)
            S = sp.array([i for i in S_K if i > 1e-9])
            U = U_K[:, -len(S) :]
            # In most cases W = E but have left it as seperate parameter for
            # flexibility
            self.W = U * S ** 0.5
            self.gp = Kron2Sum(self.y, sp.eye(1), self.F, self.W, restricted=True)
            self.gp.fit(verbose=verbose)
            #  Get optimal kernel parameters
            self.covarparam0 = self.gp.C0[0, 0]
            self.covarparam1 = self.gp.C1[0, 0]
            RV = None
        elif self.W is not None:
            self.gp = Kron2Sum(self.y, sp.eye(1), self.F, self.W, restricted=True)
            self.gp.fit(verbose=verbose)
            self.covarparam0 = self.gp.C0[0, 0]
            self.covarparam1 = self.gp.C1[0, 0]
            RV = None
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
        import scipy as sp
        import scipy.linalg as la
        import scipy.stats as st

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
            # from time import time
            # start = time()
            pvalue = optimal_davies_pvalue(
                qmin, MuQ, VarQ, KerQ, eigh, vareta, Df, tau_rho, self.rho_list, T
            )
            # print("Elapsed: {} seconds".format(time() - start))

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
