import time

from . import StructLMM

TESTS = ["interaction", "association"]


def run_structlmm(
    snps,
    bim,
    pheno,
    env,
    covs=None,
    rhos=None,
    batch_size=1000,
    tests=None,
    snp_preproc=None,
):
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
    snp_preproc : dict
        Dictionary specifying threshold for variant filtering.
        Supported filters: `min_max`, `max_miss`
        Default is None.

    Returns
    -------
    res : *:class:`pandas.DataFrame`*
        contains pv of joint test, pv of interaction test
        (if no_interaction_test is False) and snp info.
    """
    import pandas as pd
    import geno_sugar as gs
    import geno_sugar.preprocess as prep
    from sklearn.impute import SimpleImputer
    import scipy as sp

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
        slmm.fit_null(F=covs, verbose=False)

    if snp_preproc is None:
        snp_preproc = {}

    # define geno preprocessing function
    prep_steps = []
    # 1. missing values
    if "max_miss" in snp_preproc:
        max_miss = snp_preproc["max_miss"]
        assert type(max_miss) == float, "max_miss should be a float"
        assert max_miss >= 0 and max_miss <= 1, "max_miss should be in [0, 1]"
        prep_steps.append(prep.filter_by_missing(max_miss=max_miss))
    # 2. impute
    imputer = SimpleImputer(missing_values=sp.nan, strategy="mean")
    prep_steps.append(prep.impute(imputer))
    # 3. minimum maf
    if "min_maf" in snp_preproc:
        min_maf = snp_preproc["min_maf"]
        assert type(min_maf) == float, "min_maf should be a float"
        assert min_maf >= 0 and min_maf <= 1, "min_maf should be in [0, 1]"
        prep_steps.append(prep.filter_by_maf(min_maf=min_maf))
    # 2. standardize
    prep_steps.append(prep.standardize())
    preprocess = gs.preprocess.compose(prep_steps)

    t0 = time.time()

    # loop on geno
    res = []
    queue = gs.GenoQueue(snps, bim, batch_size=batch_size, preprocess=preprocess)
    for _G, _bim in queue:

        _pv = sp.zeros(_G.shape[1])
        _pv_int = sp.zeros(_G.shape[1])
        for snp in range(_G.shape[1]):
            print(snp)
            x = _G[:, [snp]]

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

    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)

    t = time.time() - t0
    print("%.2f s elapsed" % t)

    return res
