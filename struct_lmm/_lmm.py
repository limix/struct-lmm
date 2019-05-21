from chiscore import optimal_davies_pvalue, davies_pvalue


class StructLMM:
    r"""
    Structured linear mixed model that accounts for genotype-environment interactions.

    Let n be the number of samples.
    StructLMM [1] extends the conventional linear mixed model by including an
    additional per-individual effect term that accounts for genotype-environment
    interaction, which can be represented as an n×1 vector, 𝛃.
    The model is given by

        𝐲 = 𝙼𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + 𝛆,

    where

        𝛽 ∼ 𝓝(0, 𝓋₀⋅ρ), 𝛃 ∼ 𝓝(𝟎, 𝓋₀(1-ρ)𝙴𝙴ᵀ), 𝐞 ∼ 𝓝(𝟎, 𝓋₁𝚆𝚆ᵀ), and 𝛆 ∼ 𝓝(𝟎, 𝓋₂𝙸).

    The vector 𝐲 is the outcome, matrix 𝙼 contains the covariates, and vector 𝐠 is the
    genetic variant.
    The matrices 𝙴 and 𝚆 are generally the same, and represent the environment
    configuration for each sample.
    The parameters 𝓋₀, 𝓋₁, and 𝓋₂ are the overall variances.
    The parameter ρ ∈ [𝟶, 𝟷] dictates the relevance of genotype-environment interaction
    versus the genotype effect alone.
    The term 𝐞 accounts for additive environment-only effects while 𝛆 accounts for
    noise effects.

    The above model is equivalent to

        𝐲 = 𝙼𝛂 + 𝐠⊙𝛃 + 𝐞 + 𝛆,

    where

        𝛃 ∼ 𝓝(𝟎, 𝓋₀(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)), 𝐞 ∼ 𝓝(𝟎, 𝓋₁𝚆𝚆ᵀ), and 𝛆 ∼ 𝓝(𝟎, 𝓋₂𝙸).

    Its marginalised form is given by

        𝐲 ∼ 𝓝(𝙼𝛂, 𝓋₀𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳 + 𝓋₁𝚆𝚆ᵀ + 𝓋₂𝙸),

    where 𝙳 = diag(𝐠).

    StructLMM method is used to perform two types of statistical tests.
    The association one compares the following hypothesis:

        𝓗₀: 𝓋₀ = 0
        𝓗₁: 𝓋₀ > 0

    𝓗₀ denotes no genetic association, while 𝓗₁ models any genetic association.
    In particular, 𝓗₁ includes genotype-environment interaction as part of genetic
    association.
    The interaction test is slightly more complicated as the term 𝐠𝛽 is now considered
    a fixed one. In pratice, we include 𝐠 in the covariates matrix 𝙼 and set ρ = 0.
    We refer to this modified model as the interaction model.
    The compared hypothesis are:

        𝓗₀: 𝓋₀ = 0 (given the interaction model)
        𝓗₁: 𝓋₀ > 0 (given the interaction model)

    Implementation
    --------------

    We employ the score-test statistic [2] for both tests

        𝑄 = ½𝐲ᵀ𝙿(∂𝙺)𝙿𝐲,

    where

        𝙿 = 𝙺⁻¹ - 𝙺⁻¹𝙼(𝙼ᵀ𝙺⁻¹𝙼)⁻¹𝙼ᵀ𝙺⁻¹ and cov(𝐲) = 𝙺

    for the REML-estimated parameters under the null hypothesis.
    The derivative is taken over the parameter being tested.

    Lets for now assume that ρ is given.
    In practice, we have

        𝙺ᵨ  = 𝓋₀𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳 + 𝓋₁𝚆𝚆ᵀ + 𝓋₂𝙸
        ∂𝙺ᵨ = 𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳

    for association test and

        𝙺₀  = 𝓋₀𝙳𝙴𝙴ᵀ𝙳 + 𝓋₁𝚆𝚆ᵀ + 𝓋₂𝙸
        ∂𝙺₀ = 𝙳𝙴𝙴ᵀ𝙳

    for interaction test, for parameters estimated via REML.
    The outcome distribution under null is

        𝐲 ∼ 𝓝(𝙼𝛂, 𝓋₁𝚆𝚆ᵀ + 𝓋₂𝙸).

    It can be shown [2]_ that

        𝑄 ∼ ∑ᵢ𝜆ᵢ𝜒²(1),

    where the weights 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺)√𝙿.
    We employ modified Liu approximation to 𝑄 proposed [3] and modified in [4].

    References
    ----------
    .. [1] Moore, R., Casale, F. P., Bonder, M. J., Horta, D., Franke, L., Barroso,
       I., & Stegle, O. (2018). A linear mixed-model approach to study multivariate
       gene–environment interactions (p. 1). Nature Publishing Group.
    .. [2] Lippert, C., Xiang, J., Horta, D., Widmer, C., Kadie, C., Heckerman, D.,
       & Listgarten, J. (2014). Greater power and computational efficiency for
       kernel-based association testing of sets of genetic variants. Bioinformatics,
       30(22), 3206-3214.
    .. [3] Liu, H., Tang, Y., & Zhang, H. H. (2009). A new chi-square approximation
       to the distribution of non-negative definite quadratic forms in non-central
       normal variables. Computational Statistics & Data Analysis, 53(4), 853-856.
    .. [4] Lee, Seunggeun, Michael C. Wu, and Xihong Lin. "Optimal tests for rare
       variant effects in sequencing association studies." Biostatistics 13.4 (2012):
       762-775.
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
        self._rhos = [0.0, 0.1 ** 2, 0.2 ** 2, 0.3 ** 2, 0.4 ** 2, 0.5 ** 2, 0.5, 0.999]

    def fit(self, verbose=True):
        from glimix_core.lmm import Kron2Sum

        self._lmm = Kron2Sum(self._y, [[1]], self._M, self._W, restricted=True)
        self._lmm.fit(verbose=verbose)
        self._covarparam0 = self._lmm.C0[0, 0]
        self._covarparam1 = self._lmm.C1[0, 0]

    def _P(self, v):
        """
        Let 𝙺 be the optimal covariance matrix under the null hypothesis.
        Given 𝐯, this method computes

            𝙿𝐯 = 𝙺⁻¹𝐯 - 𝙺⁻¹𝙼(𝙼ᵀ𝙺⁻¹𝙼)⁻¹𝙼ᵀ𝙺⁻¹𝐯.
        """
        from numpy_sugar.linalg import rsolve
        from scipy.linalg import cho_solve

        x = rsolve(self._lmm.covariance(), v)
        if self._lmm.X is not None:
            Lh = self._lmm._terms["Lh"]
            t = self._lmm.X @ cho_solve(Lh, self._lmm.M.T @ x)
            x -= rsolve(self._lmm.covariance(), t)

        return x

    def _score_stats(self, g, rhos):
        """
        Let 𝙺 be the optimal covariance matrix under the null hypothesis.
        For a given ρ, the score-based test statistic is given by

            𝑄ᵨ = ½𝐲ᵀ𝙿ᵨ(∂𝙺ᵨ)𝙿ᵨ𝐲,

        where

            ∂𝙺ᵨ = 𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳

        and 𝙳 = diag(𝐠).
        """
        from numpy_sugar import ddot
        from numpy import zeros

        Q = zeros(len(rhos))
        DPy = ddot(g, self._P(self._y))
        s = DPy.sum()
        l = s * s
        DPyE = DPy.T @ self._E
        r = DPyE @ DPyE.T
        for i, rho in enumerate(rhos):
            Q[i] = (rho * l + (1 - rho) * r) / 2

        return Q

    def _score_stats_null_dist(self, g):
        """
        Under the null hypothesis, the score-based test statistic follows a weighted sum
        of random variables:

            𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1),

        where 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺)√𝙿.

        Note that

            ∂𝙺ᵨ = 𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳 = (ρ𝐠𝐠ᵀ + (1-ρ)𝙴̃𝙴̃ᵀ)

        for 𝙴̃ = 𝙳𝙴.
        By using SVD decomposition, one can show that the non-zero eigenvalues of 𝚇𝚇ᵀ
        are equal to the non-zero eigenvalues of 𝚇ᵀ𝚇.
        Therefore, 𝜆ᵢ are the non-zero eigenvalues of

            ½[√ρ𝐠 √(1-ρ)𝙴̃]𝙿[√ρ𝐠 √(1-ρ)𝙴̃]ᵀ.

        """
        from numpy import empty
        from numpy.linalg import eigvalsh
        from math import sqrt
        from numpy_sugar import ddot

        Et = ddot(g, self._E)
        Pg = self._P(g)
        PEt = self._P(Et)

        gPg = g.T @ Pg
        EtPEt = Et.T @ PEt
        gPEt = g.T @ PEt

        n = Et.shape[1] + 1
        F = empty((n, n))

        lambdas = []
        for i in range(len(self._rhos)):
            rho = self._rhos[i]

            F[0, 0] = rho * gPg
            F[0, 1:] = sqrt(rho) * sqrt(1 - rho) * gPEt
            F[1:, 0] = F[0, 1:]
            F[1:, 1:] = (1 - rho) * EtPEt

            lambdas.append(eigvalsh(F) / 2)

        return lambdas

    def _score_stats_pvalue(self, Qs, lambdas):
        """
        Computes Pr(𝑄 > q) for 𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1).

        Pr(𝑄 > q) is the p-value for the score statistic.

        Parameters
        ----------
        Qs : array_like
            𝑄ᵨ statistic.
        lambdas : array_like
            𝜆ᵢ from the null distribution for each ρ.
        """
        from numpy import stack

        return stack([_mod_liu(Q, lam) for Q, lam in zip(Qs, lambdas)], axis=0)

    def _qmin(self, pliumod):
        from numpy import zeros
        import scipy.stats as st

        # T statistic
        T = pliumod[:, 0].min()

        qmin = zeros(len(self._rhos))
        percentile = 1 - T
        for i in range(len(self._rhos)):
            q = st.chi2.ppf(percentile, pliumod[i, 3])
            mu_q = pliumod[i, 1]
            sigma_q = pliumod[i, 2]
            dof = pliumod[i, 3]
            qmin[i] = (q - dof) / (2 * dof) ** 0.5 * sigma_q + mu_q

        return qmin

    # SKAT
    def score_2dof_inter(self, X):
        from numpy import empty
        from numpy_sugar import ddot

        Q_rho = self._score_stats(X.ravel(), [0])

        g = X.ravel()
        Et = ddot(g, self._E)
        PEt = self._P(Et)

        EtPEt = Et.T @ PEt
        gPEt = g.T @ PEt

        n = Et.shape[1] + 1
        F = empty((n, n))

        F[0, 0] = 0
        F[0, 1:] = gPEt
        F[1:, 0] = F[0, 1:]
        F[1:, 1:] = EtPEt
        F /= 2

        return davies_pvalue(Q_rho[0], F)

    # SKAT-O
    def score_2dof_assoc(self, X):
        from numpy import trace, sum, where, empty
        from numpy.linalg import eigvalsh

        Q_rho = self._score_stats(X.ravel(), self._rhos)
        null_lambdas = self._score_stats_null_dist(X.ravel())
        pliumod = self._score_stats_pvalue(Q_rho, null_lambdas)
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
        eigh = eigvalsh(ZTIminusMZ)

        eta = ETxPx11xPxE @ ZTIminusMZ
        vareta = 4 * trace(eta)

        OneZTZE = 0.5 * (X.T @ PxoE)
        tau_top = OneZTZE @ OneZTZE.T
        tau_rho = empty(len(self._rhos))
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
        idx = where(pliumod[:, 0] > 0)[0]
        pval = pliumod[:, 0].min() * multi
        if pvalue <= 0 or len(idx) < len(self._rhos):
            pvalue = pval
        if pvalue == 0:
            if len(idx) > 0:
                pvalue = pliumod[:, 0][idx].min()

        return pvalue


def _mod_liu(q, w):
    from chiscore import liu_sf

    (pv, dof_x, _, info) = liu_sf(q, w, [1] * len(w), [0] * len(w), True)
    return (pv, info["mu_q"], info["sigma_q"], dof_x)
