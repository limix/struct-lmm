import time
import sys
import os
import numpy as np
import pandas as pd
import scipy as sp
from optparse import OptionParser
from limix_core.util.preprocess import gaussianize
from struct_lmm import run_structlmm
from struct_lmm.utils.sugar_utils import norm_env_matrix
from pandas_plink import read_plink
import geno_sugar as gs


if __name__=='__main__':

    # import genotype file
    bedfile = 'data_structlmm/chrom22_subsample20_maf0.10'
    (bim, fam, G) = read_plink(bedfile)

    # subsample snps
    Isnp = gs.is_in(bim, ('22', 17500000, 18000000))
    G, bim = gs.snp_query(G, bim, Isnp)

    # load phenotype file
    phenofile = 'data_structlmm/expr.csv'
    dfp = pd.read_csv(phenofile, index_col=0)
    pheno = gaussianize(dfp.loc['gene1'].values[:, None])

    # load environment file and normalize
    envfile = 'data_structlmm/env.txt'
    E = sp.loadtxt(envfile)
    E = norm_env_matrix(E)

    # mean as fixed effect
    covs = sp.ones((E.shape[0], 1))

    # run analysis with struct lmm
    res = run_structlmm(G, bim, pheno, E, covs=covs, batch_size=100, unique_variants=True)

    # export
    print('Export')
    if not os.path.exists('out'): os.makedirs('out')
    res.to_csv('out/res_structlmm.csv', index=False)
