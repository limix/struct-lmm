from numpy.testing import assert_allclose
from numpy.random import RandomState
import scipy as sp
from struct_lmm import StructLMM 

def test_structlmm():
    #1. generate data
    random = RandomState(1)
    n = 20
    k = 4
    y = random.randn(n)
    E = random.randn(n, k)
    rho = [0., .2, .4, .6, .8, 1.]
    covs = sp.ones((n, 1))

    #2. fit null model
    import pdb; pdb.set_trace()
    slmmm = StructLMM(y, E, W=E, rho_list=rho)
    null = hGWASjoint.fit_null(F = covs, verbose=False)

    #3. score test
    pv, rho_opt = hGWASjoint.score_2_dof(x)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])

