import os
import sys
import time
from optparse import OptionParser

import h5py
import numpy as np
import pandas as pd
import scipy as sp

from struct_lmm.utils.sugar_utils import *


def entry_point():

    parser = OptionParser()

    # input files
    parser.add_option("--in", dest='inf', type=str, default=None)
    parser.add_option("--out", dest='outf', type=str, default=None)
    parser.add_option("--norm_type", dest='norm_type', type = str, default='weighted_correlation')

    (opt, args) = parser.parse_args()

    assert opt.inf is not None, 'Specify in file!'
    assert opt.outf is not None, 'Specify out file!'

    allowed_types = ['linear_covariance', 'weighted_correlation', 'correlation']
    assert norm_type in allowed_types, 'Value of norm_type not allowed'

    E = sp.loadtxt(opt.inf)
    En = norm_env_matrix(E, norm_type=opt.norm_type)
    sp.savetxt(opt.outf, En)


if __name__ == '__main__':

    entry_point()
