import os
import sys
import time
from optparse import OptionParser
import pandas as pd
import scipy as sp
from struct_lmm import run_structlmm
from struct_lmm.utils.sugar_utils import *
from pandas_plink import read_plink
import geno_sugar as gs


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

    # size of snps to load into memory
    parser.add_option(
        "--batch_size", dest='batch_size', type=int, default=1000)

    # analysis options
    parser.add_option("--rhos", dest='rhos', type=str, default=None)
    parser.add_option(
        "--no_interaction",
        action="store_true",
        dest='no_interaction',
        default=False)
    parser.add_option(
        "--no_association",
        action="store_true",
        dest='no_association',
        default=False)
    (opt, args) = parser.parse_args()

    # assert stuff
    assert opt.bfile is not None, 'Specify bed file!'
    assert opt.pfile is not None, 'Specify pheno file!'
    assert opt.efile is not None, 'Specify env file!'
    assert opt.ofile is not None, 'Specify out file!'

    # import genotype file
    (bim, fam, G) = read_plink(opt.bfile)

    # subsample snps
    Isnp = sp.ones(bim.shape[0], dtype=bool)
    if opt.i0 is not None:      Isnp = Isnp & (bim.i.values>=opt.i0)
    if opt.i1 is not None:      Isnp = Isnp & (bim.i.values<opt.i1)
    if opt.chrom is not None:   Isnp = Isnp & (bim.chrom.values==opt.chrom)
    if opt.pos_start is not None:  Isnp = Isnp & (bim.pos.values>=opt.pos_start)
    if opt.pos_end is not None:  Isnp = Isnp & (bim.pos.values>opt.pos_end)
    G, bim = gs.snp_query(G, bim, Isnp)

    # pheno
    y = import_one_pheno_from_csv(
        opt.pfile, pheno_id=opt.pheno_id, standardize=True)

    # import environment
    E = sp.loadtxt(opt.efile)

    # import fixed effects
    if opt.ffile is None:
        covs = sp.ones((E.shape[0], 1))
    else:
        covs = sp.loadtxt(opt.ffile)

    # extract rhos
    if opt.rhos no is None: 
        rhos = sp.array(opt.rhos.split(','), dtype=float)
    else:
        rhos = sp.array([0., 0.1**2, 0.2**2, 0.3**2, 0.4**2, 0.5**2, 0.5, 1.])

    # tets
    test = None
    if opt.no_interaction:
        tests = ['association']
    if opt.no_association:
        tests = ['interaction']

    # run analysis
    res = run_struct_lmm(
        G,
        bim,
        y,
        E,
        covs=covs,
        rhos=rhos,
        batch_size=opt.batch_size,
        tests=tests,
        unique_variants=opt.unique_variants)

    # export
    print('Export to %s' % opt.ofile)
    make_out_dir(opt.ofile)
    res.to_csv(opt.ofile, index=False)


if __name__ == '__main__':

    entry_point()
