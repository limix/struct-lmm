from numpy.testing import assert_allclose

def test_mocktest():
    assert_allclose(sp.ones(3), sp.ones(3))

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])

