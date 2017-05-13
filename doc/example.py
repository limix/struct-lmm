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
from struct_lmm import run_struct_lmm 
from struct_lmm import run_lmm_int
from struct_lmm import run_lmm
from struct_lmm.utils.sugar_utils import *

if __name__=='__main__':

    # define bed, phenotype and environment files
    bedfile = 'data_structlmm/chrom22_subsample20_maf0.10'
    phenofile = 'data_structlmm/expr.csv'
    envfile = 'data_structlmm/env.txt'

    # import geno and subset to first 1000 variants
    reader = BedReader(bedfile)
    query = build_geno_query(idx_start=0, idx_end=1000)
    reader.subset_snps(query, inplace=True)

    # pheno
    y = import_one_pheno_from_csv(phenofile,
                                  pheno_id='gene1',
                                  standardize=True)

    # import environment and normalize
    E = sp.loadtxt(envfile)
    E = norm_env_matrix(E)

    # mean as fixed effect 
    covs = sp.ones((E.shape[0], 1))

    # run analysis with struct lmm
    res_slmm = run_struct_lmm(reader, y, E,
                          covs=covs,
                          batch_size=100,
                          unique_variants=True)

    # run analysis with fixed-effect lmm
    # envs are modelled as random effects
    res_int = run_lmm_int(reader, y, E,
                          W=E,
                          covs=covs,
                          batch_size=100,
                          unique_variants=True)

    # run analysis with standard lmm
    # pure environment is modelled as random effects 
    res_lmm = run_lmm(reader, y, W=E,
                      covs=covs,
                      batch_size=100,
                      unique_variants=True)

    # export
    print 'Export'
    if not os.path.exists('out'):
        os.makedirs('out')
    res_slmm.to_csv('out/res_structlmm.csv', index=False)
    res_int.to_csv('out/res_int.csv', index=False)
    res_lmm.to_csv('out/res_lmm.csv', index=False)

