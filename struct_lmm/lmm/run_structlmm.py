import time
import pandas as pd
import scipy as sp
from . import StructLMM
import geno_sugar as gs
import geno_sugar.preprocess
from sklearn.preprocessing import Imputer

TESTS = ['interaction', 'association']


def run_structlmm(snps,
                   bim,
                   pheno,
                   env,
                   covs=None,
                   rhos=None,
                   batch_size=1000,
                   tests=None,
                   unique_variants=False):
    """
    Utility function to run StructLMM

    Parameters
    ----------
    snps : array_like
        snps data
    bim : pandas.DataFrame
        snps annot
    pheno : (`N`, 1) ndarray
        phenotype vector
    env : (`N`, `K`)
          Environmental matrix (indviduals by number of environments)
    covs : (`N`, L) ndarray
        fixed effect design for covariates `N` samples and `L` covariates.
    rhos : list
        list of ``rho`` values.  Note that ``rho = 1-rho`` in the equation described above.
        ``rho=0`` correspond to no persistent effect (only GxE);
        ``rho=1`` corresponds to only persistent effect (no GxE);
        By default, ``rho=[0, 0.1**2, 0.2**2, 0.3**2, 0.4**2, 0.5**2, 0.5, 1.]``
    batch_size : int
        to minimize memory usage the analysis is run in batches.
        The number of variants loaded in a batch
        (loaded into memory at the same time).
    tests : list
        list of tests to perform.
        Each element shoudl be in ['association', 'interation'].
        By default, both tests are considered.
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
        rhos = [0.0, 0.1 ** 2, 0.2 ** 2, 0.3 ** 2, 0.4 ** 2, 0.5 ** 2, 0.5, 1.0]

    if tests is None:
        tests = TESTS

    if TESTS[0] in tests:
        slmm_int = StructLMM(pheno, env, W=env, rho_list=[0])

    if TESTS[1] in tests:
        slmm = StructLMM(pheno, env, W=env, rho_list=rhos)
        null = slmm.fit_null(F=covs, verbose=False)

    # geno preprocessing function
    impute = gs.preprocess.impute(Imputer(missing_values=sp.nan, strategy='mean', axis=1))
    standardize = gs.preprocess.standardize()
    preprocess = gs.preprocess.compose([impute, standardize])

    # filtering funciton
    filter = None
    if unique_variants:
        filter = gs.unique_variants

    t0 = time.time()

    # loop on geno
    res = []
    n_analyzed = 0
    queue = gs.GenoQueue(snps, bim, batch_size=50, preprocess=preprocess, filter=filter)
    for _G, _bim in queue:

        _pv = sp.zeros(_G.shape[0])
        _pv_int = sp.zeros(_G.shape[0])
        for snp in range(_G.shape[0]):
            x = _G[[snp], :].T

            if TESTS[0] in tests:
                # interaction test
                covs1 = sp.hstack((covs, x))
                slmm_int.fit_null(F=covs1, verbose=False)
                _p = slmm_int.score_2_dof(x)
                _pv_int[snp] = _p

            if TESTS[1] in tests:
                # association test
                _p = slmm.score_2_dof(x)
                _pv[snp] = _p

        if TESTS[0] in tests:
            _bim = _bim.assign(pv_int=pd.Series(_pv_int, index=_bim.index))

        if TESTS[1] in tests:
            _bim = _bim.assign(pv=pd.Series(_pv, index=_bim.index))

        # add pvalues to _res and append to res
        res.append(_bim)

        n_analyzed += _G.shape[0]
        print('.. analysed %d/%d variants' % (n_analyzed, snps.shape[0]))

    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)

    t = time.time() - t0
    print('%.2f s elapsed' % t)

    return res
