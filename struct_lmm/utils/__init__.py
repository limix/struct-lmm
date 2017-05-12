r"""
***********************
Heritability estimation
***********************

- :func:`.fdr_bh`
- :class:`.CompQuadFormLiu`
- :class:`.CompQuadFormDavies`
- :class:`.CompQuadFormLiuMod`
- :class:`.CompQuadFormDaviesSkat`

Public interface
^^^^^^^^^^^^^^^^
"""

from .fdr import fdr_bh
from .pvmixchi2 import CompQuadFormLiu
from .pvmixchi2 import CompQuadFormDavies
from .pvmixchi2 import CompQuadFormLiuMod
from .pvmixchi2 import CompQuadFormDaviesSkat

__all__ = ['fdr', 'CompQuadFormLiu', 'CompQuadFormDavies',
           'CompQuadFormLiuMod', 'CompQuadFormDaviesSkat']
