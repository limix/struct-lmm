import sphinx_rtd_theme


def _get_version():
    import struct_lmm

    return struct_lmm.__version__


def _get_name():
    import struct_lmm

    return struct_lmm.__name__


extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

napoleon_numpy_docstring = True
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
project = _get_name()
copyright = "2017, Danilo Horta, Francesco Paolo Casale, Oliver Stegle, Rachel Moore"
author = "Danilo Horta, Francesco Paolo Casale, Oliver Stegle, Rachel Moore"
version = _get_version()
release = version
language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py"]
pygments_style = "default"
todo_include_todos = False
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = "{}doc".format(project)
intersphinx_mapping = {"https://docs.python.org/": None}
