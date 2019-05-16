import scipy as sp
from numpy.random import RandomState
from numpy.testing import assert_allclose

from struct_lmm import StructLMM


def test_structlmm():
    random = RandomState(1)
    n = 20
    k = 4
    y = random.randn(n, 1)
    E = random.randn(n, k)
    covs = sp.ones((n, 1))
    x = 1.0 * (random.rand(n, 1) < 0.2)

    slmm = StructLMM(y, covs, E, W=E)
    slmm.fit(verbose=False)

    pv = slmm.score_2_dof(x)
    assert_allclose([pv], [0.7066731768614625], rtol=1e-6)
