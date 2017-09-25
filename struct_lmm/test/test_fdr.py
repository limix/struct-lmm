from numpy import array
from numpy.testing import assert_allclose

from struct_lmm.utils import fdr_bh


def test_fdr():
    pv = array([.1, .5, .9])
    qv = fdr_bh(pv)
    assert_allclose(qv, [0.3, 0.75, 0.9])


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
