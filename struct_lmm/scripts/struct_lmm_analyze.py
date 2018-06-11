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
from limix.util.preprocess import regressOut, gaussianize
from limix.util import unique_variants
from struct_lmm.runner import run_struct_lmm
from struct_lmm.lmm import LMM
from struct_lmm.utils.sugar_utils import *


def entry_point():

    parser = OptionParser()

    # input files
    parser.add_option("--bfile", dest='bfile', type=str, default=None)
    parser.add_option("--pfile", dest='pfile', type=str, default=None)
    parser.add_option("--efile", dest='efile', type=str, default=None)
    parser.add_option("--ffile", dest='ffile', type=str, default=None)

    # output file
    parser.add_option("--ofile", dest='ofile', type=str, default=None)

    # phenotype filtering
    parser.add_option("--pheno_id", dest='pheno_id', type=str, default=None)

    # snp filtering options
    parser.add_option("--idx_start", dest='i0', type=int, default=None)
    parser.add_option("--idx_end", dest='i1', type=int, default=None)
    parser.add_option("--chrom", dest='chrom', type=int, default=None)
    parser.add_option("--pos_start", dest='pos_start', type=int, default=None)
    parser.add_option("--pos_end", dest='pos_end', type=int, default=None)

    # conditioning options and filt out
    parser.add_option("--pos_cond", dest='posc', type=int, default=None)
    parser.add_option("--nstds", dest='nstds', type=float, default=None)

    # size of batches to load into memory
    parser.add_option(
        "--batch_size", dest='batch_size', type=int, default=1000)

    # analysis options
    parser.add_option("--rhos", dest='rhos', type=str, default=None)
    parser.add_option(
        "--unique_variants",
        action="store_true",
        dest='unique_variants',
        default=False)
    parser.add_option(
        "--no_interaction_test",
        action="store_true",
        dest='no_interaction_test',
        default=False)
    (opt, args) = parser.parse_args()

    # assert stuff
    assert opt.bfile is not None, 'Specify bed file!'
    assert opt.pfile is not None, 'Specify pheno file!'
    assert opt.efile is not None, 'Specify env file!'
    assert opt.ofile is not None, 'Specify out file!'
    if opt.rhos is None: opt.rhos = '0.,.2,.4,.6,.8,1.'

    # import geno and subset
    reader = BedReader(opt.bfile)
    query = build_geno_query(
        idx_start=opt.i0,
        idx_end=opt.i1,
        chrom=opt.chrom,
        pos_start=opt.pos_start,
        pos_end=opt.pos_end)
    reader.subset_snps(query, inplace=True)

    # pheno
    y = import_one_pheno_from_csv(
        opt.pfile, pheno_id=opt.pheno_id, standardize=True)

    # import environment
    E = sp.loadtxt(opt.efile)

    # regress out conditioning genotype
    if opt.posc:
        query = build_geno_query(
            chrom=opt.chrom,
            pos_start=opt.posc,
            pos_end=opt.posc+1)
        xc = reader.getGenotypes(query, impute=True)
        # comp pvalues
        covs = sp.ones_like(xc)
        pv_env = []
        for ie in range(E.shape[1]):
            lmm = LMM(E[:,[ie]], covs)
            lmm.process(xc)
            pv_env.append(lmm.getPv()[0])
        pv_env = sp.array([sp.array(pv_env).min()])
        # regressout
        _W = sp.concatenate([xc, covs], 1)
        E = regressOut(E, _W)
        E = gaussianize(E)

    # import fixed effects
    if opt.ffile is None:
        covs = sp.ones((E.shape[0], 1))
    else:
        covs = sp.loadtxt(opt.ffile)

    # extract rhos
    rhos = sp.array(opt.rhos.split(','), dtype=float)

    Isample = None
    if opt.nstds is not None:
        Isample = sp.logical_and(y[:,0]<opt.nstds, y[:,0]>-opt.nstds)

    # run analysis
    res = run_struct_lmm(
        reader,
        y,
        E,
        covs=covs,
        rhos=rhos,
        batch_size=opt.batch_size,
        no_interaction_test=opt.no_interaction_test,
        unique_variants=opt.unique_variants,
        Isample=Isample)

    if opt.posc is not None:
        res = res.assign(pv_env=pd.Series(pv_env, index=res.index))

    # export
    print 'Export to %s' % opt.ofile
    make_out_dir(opt.ofile)
    res.to_csv(opt.ofile, index=False)


if __name__ == '__main__':

    entry_point()
