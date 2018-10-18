r"""
*********
StructLMM
*********

Write me

"""

from __future__ import absolute_import as _

from . import interpretation, lmm, utils
from .lmm import StructLMM, run_structlmm
from ._testit import test

__version__ = "0.1.0"

__all__ = ['lmm', 'interpretation', 'utils', '__version__', "test"]
