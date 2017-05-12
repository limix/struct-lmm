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
from struct_lmm import StructLMM

if __name__=='__main__':
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

    # size of batches to load into memory
    parser.add_option("--batch_size", dest='batch_size', type=int, default=1000)

    # analysis options
    parser.add_option("--rhos", dest='rhos', type=str, default=None)
    parser.add_option("--no_mean_to_one",
                      action="store_true",
                      dest='no_mean_to_one',
                      default=False)
    parser.add_option("--unique_variants",
                      action="store_true",
                      dest='unique_variants',
                      default=False)
    parser.add_option("--no_interaction_test",
                      action="store_true",
                      dest='no_interaction_test',
                      default=False)
    (opt, args) = parser.parse_args()

    # geno
    assert opt.bfile is not None, 'Specify bed file!'
    reader = BedReader(opt.bfile)
    query = build_geno_query(idx_start=opt.i0,
                             idx_end=opt.i1,
                             chrom=opt.chrom,
                             pos_start=opt.pos_start,
                             pos_end=opt.pos_end)
    reader.subset_snps(query, inplace=True)
    n_snps = reader.getSnpInfo().shape[0]
    n_batches = int(sp.ceil(n_snps/float(opt.batch_size)))

    # pheno
    assert opt.pfile is not None, 'Specify pheno file!'
    #df = dd.read_csv('data/expr_table.csv', header=None)
    df2 = dd.read_csv(opt.pfile)
    Ip = df2['Unnamed: 0']==opt.pheno_id
    del df2['Unnamed: 0']
    y = df2[Ip].values.compute().T
    y -= y.mean(0)
    y /= y.std(0)

    assert opt.efile is not None, 'Specify env file!'
    E = sp.loadtxt(opt.efile)
    if opt.no_mean_to_one:
        E *= sp.sqrt(E.shape[0] / sp.sum(E**2))
    else:
        E /= ((E**2).sum(1)**0.5)[:, sp.newaxis]

    if opt.ffile is None:
        covs = sp.ones((E.shape[0], 1))
    else:
        covs = sp.loadtxt(opt.ffile)

    assert opt.ofile is not None, 'Specify out file!'
    resdir = '/'.join(sp.array(opt.efile.split('/'))[:-1])
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    if opt.rhos is None:
        opt.rhos = '0.,.2,.4,.6,.8,1.'
    rhos = sp.array(opt.rhos.split(','), dtype=float)

    # slmm fit null 
    slmm = StructLMM(y, E, W=E, rho_list=rhos)
    null = slmm.fit_null(F=covs, verbose=False)

    # slmm int
    if not opt.no_interaction_test:
        slmm_int = StructLMM(y, E, W=E, rho_list=[0])

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

