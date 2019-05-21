from numpy import ones, stack
from numpy.random import RandomState
from numpy.testing import assert_allclose

from struct_lmm import StructLMM


def test_structlmm_assoc():
    random = RandomState(1)
    n = 20
    k = 4
    y = random.randn(n, 1)
    E = random.randn(n, k)
    M = ones((n, 1))
    x = 1.0 * (random.rand(n, 1) < 0.2)

    slmm = StructLMM(y, M, E, W=E)
    slmm.fit(verbose=False)

    pv = slmm.score_2dof_assoc(x)
    assert_allclose([pv], [0.8470039620073695], rtol=1e-5)


def test_structlmm_inter():
    random = RandomState(1)
    n = 20
    k = 4
    y = random.randn(n, 1)
    E = random.randn(n, k)
    M = ones(n)
    x = 1.0 * (random.rand(n) < 0.2)
    M = stack([M, x], axis=1)

    slmm = StructLMM(y, M, E, W=E)
    slmm.fit(verbose=False)

    pv = slmm.score_2dof_inter(x)
    assert_allclose([pv], [0.6781070640353783], rtol=1e-5)
