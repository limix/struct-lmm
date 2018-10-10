import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests


def fdr_bh(pv):
    return multipletests(np.asarray(pv), method="fdr_bh")[1]
