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
    parser.add_option(
        "--no_mean_to_one",
        action="store_true",
        dest='no_mean_to_one',
        default=False)
    (opt, args) = parser.parse_args()

    assert opt.inf is not None, 'Specify in file!'
    assert opt.outf is not None, 'Specify out file!'

    E = sp.loadtxt(opt.inf)
    En = norm_env_matrix(E, no_mean_to_one=opt.no_mean_to_one)
    sp.savetxt(opt.outf, En)


if __name__ == '__main__':

    entry_point()
