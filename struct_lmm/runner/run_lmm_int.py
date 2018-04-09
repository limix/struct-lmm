import os
import sys
import time
import warnings
from optparse import OptionParser

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as la
import scipy.stats as st

from limix.data import BedReader, GIter, build_geno_query
from limix.util import unique_variants as f_univar
from struct_lmm.lmm import LMM, LMMCore
from struct_lmm.utils.sugar_utils import *


def run_lmm_int(reader,
                pheno,
                env,
                R=None,
                S_R=None,
                U_R=None,
                W=None,
                covs=None,
                batch_size=1000,
                unique_variants=False):
    """
    Utility function to run interaction tests.

    Parameters
    ----------
    reader : :class:`limix.data.BedReader`
        limix bed reader instance.
    pheno : (`N`, 1) ndarray
        phenotype vector
    env : (`N`, `K`)
        Environmental matrix
        (indviduals by number of environments)
    R : (`N`, `N`) ndarray
        covariance of the random effect.
        Typically this is the genetic relatedness matrix.
        If set, ``W``, ``S_R`` and ``U_R`` are ignored.
    S_R : (`N`, ) ndarray
        eigenvalues of ``R``. If available together with the eigenvectors
        ``U_R``, they can be provided instead of ``R`` to avoid
        repeated computations.
        Only used when ``R`` is not set.
        If set, ``U_R`` should also be specified.
    U_R : (`N`, `N`) ndarray
        eigenvectors of ``R``. If available together with the eigenvalues
        ``S_R``, they can be provided instead of ``R`` to avoid
        repeated computations.
        Only used when ``R`` is not set.
        If set, ``S_R`` should also be specified.
    W : (`N`, `K`) ndarray
        this defines the covariance of a lowrank random effect.
        Setting ``W`` is equivalent to setting ``R = dot(W, W.T)``
        but ``R`` is never computed to minimize memory usage.
        Only used when ``R``, ``U_R`` and ``S_R`` are not set.
    covs : (`N`, L) ndarray
        fixed effect design for covariates `N` samples and `L` covariates.
        If None (dafault value), an intercept only is considered.
    batch_size : int
        to minimize memory usage the analysis is run in batches.
        The number of variants loaded in a batch
        (loaded into memory at the same time).
    no_interaction_test : bool
        if True the interaction test is not consdered.
        Teh default value is True.
    unique_variants : bool
        if True, only non-repeated genotypes are considered
        The default value is False.

    Returns
    -------
    res : *:class:`pandas.DataFrame`*
        contains pv, effect size, standard error on effect size,
        and test statistcs as well as variant info.
    """
    if covs is None:
        covs = sp.ones((pheno.shape[0], 1))

    # calc S_R, U_R if R is specified
    if R is not None:
        S_R, U_R = la.eigh(R)

    # assert that S_R and U_R are both specified
    S_is = S_R is not None
    U_is = U_R is not None
    if S_is or U_is:
        assert S_is and U_is, 'Both U_R and S_R should be specified'

    # assert semidefinite positiveness
    if S_R is not None:
        if S_R.min() < 1e-4:
            offset = S_R.min() + 1e-4
            S_R += offset
            warnings.warn("Added %.2e jitter to make R a SDP cov" % offset)

    # fit null
    if R is not None:
        from limix_core.gp import GP2KronSum
        from limix_core.covar import FreeFormCov
        Cg = FreeFormCov(1)
        Cn = FreeFormCov(1)
        gp = GP2KronSum(
            Y=pheno, Cg=Cg, Cn=Cn, F=covs, A=sp.eye(1), S_R=S_R, U_R=U_R)
        Cg.setCovariance(0.5 * sp.ones(1, 1))
        Cn.setCovariance(0.5 * sp.ones(1, 1))
        info_opt = gp.optimize(verbose=False)
        covar = gp.covar
    elif W is not None:
        from limix_core.gp import GP2KronSumLR
        from limix_core.covar import FreeFormCov
        gp = GP2KronSumLR(Y=pheno, Cn=FreeFormCov(1), G=W, F=covs, A=sp.eye(1))
        gp.covar.Cr.setCovariance(0.5 * sp.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * sp.ones((1, 1)))
        info_opt = gp.optimize(verbose=False)
        covar = gp.covar
    else:
        covar = None

    # define lmms
    lmm_nul = LMM(pheno, covs, covar)
    lmm_alt = LMMCore(pheno, covs, covar)

    # define inter
    ones = sp.ones([env.shape[0], 1])
    inter = sp.concatenate([ones, env], 1)

    t0 = time.time()

    res = []
    n_batches = reader.getSnpInfo().shape[0] / batch_size
    for i, gr in enumerate(GIter(reader, batch_size=batch_size)):
        print '.. batch %d/%d' % (i, n_batches)

        X, _res = gr.getGenotypes(standardize=True, return_snpinfo=True)

        if unique_variants:
            X, idxs = f_univar(X, return_idxs=True)
            Isnp = sp.in1d(sp.arange(_res.shape[0]), idxs)
            _res = _res[Isnp]

        # run lmm
        lmm_nul.process(X)
        lmm_alt.process(X, Inter=inter)

        rv = {}

        # null quantities
        rv['pv_null'] = lmm_nul.getPv()
        rv['beta_null'] = lmm_nul.getBetaSNP()
        #rv['beta_ste_null'] = lmm_nul.getBetaSNPste()
        rv['lrt_null'] = lmm_nul.getLRT()

        # alt quantities
        rv['pv_alt'] = lmm_alt.getPv()
        #rv['beta_alt'] = lmm_alt.getBetaSNP()
        #rv['beta_ste_alt'] = lmm_alt.getBetaSNPste()
        rv['lrt_alt'] = lmm_alt.getLRT()

        # inter quantities
        dof_int = inter.shape[1] - 1
        rv['lrt_int'] = rv['lrt_alt'] - rv['lrt_null']
        rv['pv_int'] = st.chi2(dof_int).sf(rv['lrt_int'])

        # add pvalues, beta, etc to res
        _res = _res.join(pd.DataFrame.from_dict(rv))
        res.append(_res)

    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)

    t = time.time() - t0
    print '%.2f s elapsed' % t

    return res
