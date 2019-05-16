"""
StructLMM
=========

Structured Linear Mixed Model is a method to test for loci that interact with multiple
environments.
"""
from ._lmm import StructLMM
from ._fdr import fdr_bh
from ._testit import test

__version__ = "0.3.0"

__all__ = ["StructLMM", "__version__", "test", "fdr_bh"]
