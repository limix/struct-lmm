import time
import sys
import os
import numpy as np
import pandas as pd
import scipy as sp
import h5py
import dask.dataframe as dd
from limix.data import BedReader
from limix.data import build_geno_query
from limix.data import GIter
from limix.util import unique_variants
from optparse import OptionParser
from struct_lmm.lmm import StructLMM

def run_struct_lmm(pheno,
                   greader,
                   E,
                   covs,
                   batch_size):

    n_batches = greader.getSnpInfo().shape[0]/batch_size

    t0 = time.time()

    pv = []
    pv_int = [] 
    for i, gr in enumerate(GIter(reader, batch_size=opt.batch_size)):
        print '.. batch %d/%d' % (i, n_batches)

        X = gr.getGenotypes(standardize=True)

        if opt.unique_variants:
            X = unique_variants(X)

        _pv = sp.zeros(X.shape[1])
        _pv_int = sp.zeros(X.shape[1])
        for snp in xrange(X.shape[1]):
            x = X[:, [snp]]

            # association test
            _p, _ = slmm.score_2_dof(x)
            _pv[snp] = _p

            if not opt.no_interaction_test:
                # interaction test
                covs1 = sp.hstack((covs, x))
                null = slmm_int.fit_null(F=covs1, verbose=False)
                _p, _ = slmm.score_2_dof(x)
                _pv[snp] = _p

    t = time.time() - t0
    print '%.2f s elapsed'

    return 

