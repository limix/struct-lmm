import scipy as sp
from numpy.random import RandomState
from numpy.testing import assert_allclose

from struct_lmm.lmm import StructLMM


def test_structlmm():
    #1. generate data
    random = RandomState(1)
    n = 20
    k = 4
    y = random.randn(n, 1)
    E = random.randn(n, k)
    rho = [0., .2, .4, .6, .8, 1.]
    covs = sp.ones((n, 1))
    x = 1. * (random.rand(n, 1) < 0.2)

    #2. fit null model
    slmm = StructLMM(y, E, W=E, rho_list=rho)
    null = slmm.fit_null(F=covs, verbose=False)

    #3. score test
    pv, rho_opt = slmm.score_2_dof(x)

    #4. assert close
    assert_allclose([pv, rho_opt], [0.83718061744700234, 0.0])


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
