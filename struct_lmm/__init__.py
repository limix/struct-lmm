"""
StructLMM
=========

"""
# from . import lmm, utils
from ._lmm import StructLMM

# from .lmm import run_structlmm
from ._testit import test

__version__ = "0.3.0"

__all__ = [
    # "lmm",
    "StructLMM",
    # "interpretation",
    # "utils",
    # "run_structlmm",
    "__version__",
    "test",
]
