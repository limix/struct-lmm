*******
Install
*******

StructLMM requires scipy, numpy, rpy2, limix-core and limix.

* Create a new environment in [conda](https://conda.io/docs/index.html)::

    conda create -n struct-lmm python=2.7
    source activate struct-lmm

* install numpy, scipy, rpy2 and limix::

    conda install -n struct-lmm numpy scipy ipython cython
    conda install -c r -n struct-lmm rpy2
    conda install -c conda-forge liknorm-py
    pip install limix

* install struct-lmm (hopefully all tests pass)::

    git clone https://github.com/limix/struct-lmm.git
    cd struct-lmm
    python setup.py install test

* install documentation::

    cd doc
    make html

Now you are all set up and ready for a :ref:`python`.
