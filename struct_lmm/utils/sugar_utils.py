import os

import dask.dataframe as dd
import scipy as sp


def import_one_pheno_from_csv(pfile, pheno_id, standardize=False):
    """
    Utility to import phenos

    Parameters
    ----------
    pfile : str
        csv file name. The csv should have row and col readers.
        See example at http://www.ebi.ac.uk/~casale/example_data/expr.csv.
        The file should contain no missing data.
    pheno_id : array-like
        phenotype to extract.
    standardize : bool (optional)
        if True the phenotype is standardized.
        The default value is False.

    Returns
    -------
    y : (`N`, `1`) array
        phenotype vactor
    """
    # read and extract
    df2 = dd.read_csv(pfile)
    key = df2.columns[0]
    Ip = df2[key] == pheno_id
    del df2[key]
    y = df2[Ip].values.compute().T

    assert not sp.isnan(y).any(), "Contains missing data!"

    if standardize:
        y -= y.mean(0)
        y /= y.std(0)

    return y


def norm_env_matrix(E, norm_type="linear_covariance"):
    """
    Normalises the environmental matrix.

    Parameters
    ----------
    E : array
        matrix of environments
    norm_type : string
        if 'linear_covariance', the environment matrix is normalized in such
        a way that the outer product EE^T has mean of diagonal of ones.
        if 'weighted_covariance', the environment matrix is normalized in such
        a way that the outer product EE^T has diagonal of ones.
        if 'correlation', the environment is normalized in such a way that the 
        outer product EE^T is a correlation matrix (with a diagonal of ones).
    Returns
    -------
    E : array
        normalised environments.
    """
    std = E.std(0)
    E = E[:, std > 0]
    E -= E.mean(0)
    E /= E.std(0)
    if norm_type == "linear_covariance":
        E *= sp.sqrt(E.shape[0] / sp.sum(E ** 2))
    elif norm_type == "weighted_covariance":
        E /= ((E ** 2).sum(1) ** 0.5)[:, sp.newaxis]
    elif norm_type == "correlation":
        E -= E.mean(1)[:, sp.newaxis]
        E /= ((E ** 2).sum(1) ** 0.5)[:, sp.newaxis]
    return E


def make_out_dir(outfile):
    """
    Util function to make out dir given an out file name.

    Parameters
    ----------
    outfile : str
        output file
    """
    resdir = "/".join(sp.array(outfile.split("/"))[:-1])
    if not os.path.exists(resdir):
        os.makedirs(resdir)
