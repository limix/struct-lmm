r"""
*********
StructLMM
*********

Write me

"""
from . import lmm, utils
from .lmm import StructLMM, run_structlmm
from ._testit import test

__version__ = "0.2.4"

__all__ = [
    "lmm",
    "StructLMM",
    "interpretation",
    "utils",
    "run_structlmm",
    "__version__",
    "test",
]
