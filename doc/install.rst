*******
Install
*******

Automatic
^^^^^^^^^

For Linux and macOS operating systems, struct-lmm can be install from the
command line by entering

.. code-block:: bash

    bash <(curl -fsSL https://raw.githubusercontent.com/limix/struct-lmm/master/install)

It will request the installation of conda_ package manager if
not already present.

Manual
^^^^^^

StructLMM requires scipy, numpy, rpy2, limix-core, and limix, among other
Python packages.
It also requires a working environment for the R programming language.
We will show here step-by-step  how to install the dependencies and StructLMM
itself.

* Create a new environment in conda_::

    conda create -n struct-lmm python=2.7
    source activate struct-lmm

* Install limix and R dependencies::

    conda install -c conda-forge limix r r-base r-essentials rpy2 r-compquadform sphinx sphinx_rtd_theme

* Install struct-lmm::

    git clone https://github.com/limix/struct-lmm.git
    cd struct-lmm
    pip install .

* Build the documentation::

    cd doc
    make html

The documentation is in HTML and will be available at
``_build/html/index.html``.

.. _conda: https://conda.io
