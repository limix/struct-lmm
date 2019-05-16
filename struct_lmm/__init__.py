"""
StructLMM
=========

"""
from ._lmm import StructLMM
from ._fdr import fdr_bh
from ._testit import test

__version__ = "0.3.0"

__all__ = ["StructLMM", "__version__", "test", "fdr_bh"]
