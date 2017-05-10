from __future__ import unicode_literals

import os
import sphinx_rtd_theme

#try:
#    import struct_lmm
#    version = struct_lmm.__version__
#except ImportError:
#    version = 'unknown'

version = 'unknown'

extensions = [
    'matplotlib.sphinxext.only_directives',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'StructLMM'
copyright = '2017, Rachel Moore, Francesco Paolo Casale, Oliver Stegle'
author = 'Rachel Moore, Francesco Paolo Casale, Oliver Stegle'
release = version
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'conf.py']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = 'struclmmdoc'
latex_elements = {}
latex_documents = [
    (master_doc, 'struct_lmm.tex', 'StructLMM Documentation',
     'Rachel Moore, Francesco Paolo Casale, Oliver Stegle',
     'manual'),
]
man_pages = [(master_doc, 'struct_lmm', 'StructLMM Documentation', [author], 1)]
texinfo_documents = [
    (master_doc, 'structLMM', 'StructLMM Documentation', author, 'structLMM',
     'A mixed-model approach to model complex GxE signals.', 'Miscellaneous'),
]
intersphinx_mapping = {'https://docs.python.org/': None,
                       'http://matplotlib.org': None}
