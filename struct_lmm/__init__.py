r"""
*********
StructLMM
*********

Write me

"""

from __future__ import absolute_import as _

from . import lmm, utils
from .lmm import StructLMM, run_structlmm
from ._testit import test

__version__ = "0.2.1"

__all__ = [
    "lmm",
    "StructLMM",
    "interpretation",
    "utils",
    "run_structlmm",
    "__version__",
    "test",
]
