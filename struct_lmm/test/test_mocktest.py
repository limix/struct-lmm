from numpy.testing import assert_allclose
from numpy import ones

def test_mocktest():
    assert_allclose(ones(3), ones(3))

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])

