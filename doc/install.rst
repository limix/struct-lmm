*******
Install
*******

StructLMM requires scipy, numpy, rpy2, limix-core and limix.

* Create a new environment in [conda](https://conda.io/docs/index.html)::

    conda create -n struct-lmm python=2.7
    source activate struct-lmm

* install limix and r dependencies::

    conda install -c conda-forge limix r r-base r-essentials rpy2 r-compquadform sphinx sphinx_rtd_theme

* install struct-lmm (hopefully all tests pass)::

    git clone https://github.com/limix/struct-lmm.git
    cd struct-lmm
    python setup.py install

* install sphinx, documentation::
    cd doc
    make html

Now you are all set up and ready for a :ref:`python`.
