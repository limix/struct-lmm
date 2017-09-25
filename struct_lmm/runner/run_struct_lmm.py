import os
import sys
import time
from optparse import OptionParser

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import scipy as sp

from limix.data import BedReader, GIter, build_geno_query
from limix.util import unique_variants as f_univar
from struct_lmm.lmm import StructLMM
from struct_lmm.utils.sugar_utils import *


def run_struct_lmm(reader,
                   pheno,
                   env,
                   covs=None,
                   rhos=None,
                   no_mean_to_one=False,
                   batch_size=1000,
                   no_association_test=False,
                   no_interaction_test=False,
                   unique_variants=False):
    """
    Utility function to run StructLMM

    Parameters
    ----------
    reader : :class:`limix.data.BedReader`
        limix bed reader instance.
    pheno : (`N`, 1) ndarray
        phenotype vector
    env : (`N`, `K`)
          Environmental matrix (indviduals by number of environments)
    covs : (`N`, L) ndarray
        fixed effect design for covariates `N` samples and `L` covariates.
    rhos : list
        list of ``rho`` values.
        ``rho=0`` correspond to no persistent effect (only GxE);
        ``rho=1`` corresponds to only persitent effect (no GxE);
        By default, ``rho=[0, 0.2, 0.4, 0.6, 0.8, 1.]``
    batch_size : int
        to minimize memory usage the analysis is run in batches.
        The number of variants loaded in a batch
        (loaded into memory at the same time).
    no_association_test : bool
        if True the association test is not consdered.
        The default value is False.
    no_interaction_test : bool
        if True the interaction test is not consdered.
        Teh default value is False.
    unique_variants : bool
        if True, only non-repeated genotypes are considered
        The default value is False.

    Returns
    -------
    res : *:class:`pandas.DataFrame`*
        contains pv of joint test, pv of interaction test
        (if no_interaction_test is False) and snp info.
    """
    if covs is None:
        covs = sp.ones((env.shape[0], 1))

    if rhos is None:
        rhos = [0, .2, .4, .6, .8, 1.]

    if not no_association_test:
        # slmm fit null
        slmm = StructLMM(pheno, env, W=env, rho_list=rhos)
        null = slmm.fit_null(F=covs, verbose=False)
    if not no_interaction_test:
        # slmm int
        slmm_int = StructLMM(pheno, env, W=env, rho_list=[0])

    n_batches = reader.getSnpInfo().shape[0] / batch_size

    t0 = time.time()

    res = []
    for i, gr in enumerate(GIter(reader, batch_size=batch_size)):
        print '.. batch %d/%d' % (i, n_batches)

        X, _res = gr.getGenotypes(standardize=True, return_snpinfo=True)

        if unique_variants:
            X, idxs = f_univar(X, return_idxs=True)
            Isnp = sp.in1d(sp.arange(_res.shape[0]), idxs)
            _res = _res[Isnp]

        _pv = sp.zeros(X.shape[1])
        _pv_int = sp.zeros(X.shape[1])
        for snp in xrange(X.shape[1]):
            x = X[:, [snp]]

            if not no_association_test:
                # association test
                _p, _ = slmm.score_2_dof(x)
                _pv[snp] = _p

            if not no_interaction_test:
                # interaction test
                covs1 = sp.hstack((covs, x))
                null = slmm_int.fit_null(F=covs1, verbose=False)
                _p, _ = slmm_int.score_2_dof(x)
                _pv_int[snp] = _p

        # add pvalues to _res and append to res
        if not no_association_test:
            _res = _res.assign(pv=pd.Series(_pv, index=_res.index))
        if not no_interaction_test:
            _res = _res.assign(pv_int=pd.Series(_pv_int, index=_res.index))
        res.append(_res)

    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)

    t = time.time() - t0
    print '%.2f s elapsed' % t

    return res
