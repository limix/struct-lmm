r"""
*********
StructLMM
*********

Write me

"""

from __future__ import absolute_import as _

from . import interpretation, lmm, runner, utils
from .lmm import LMM, LMMCore, StructLMM
from .runner import run_lmm, run_lmm_int, run_struct_lmm

__version__ = '0.0.16'

__all__ = ['lmm', 'runner', 'interpretation', 'utils', '__version__']
