*******
Install
*******

StructLMM requires scipy, numpy, rpy2, limix-core and limix.

- Create a new environment in [conda](https://conda.io/docs/index.html)::

  conda create -n struct-lmm python=2.7
  source activate struct-lmm

- install numpy, scipy, rpy2 and limix-core::

  conda install -n struct-lmm numpy scipy
  conda install -c r -n struct-lmm rpy2
  pip install limix-core

- install struct-lmm (hopefully all tests pass)::

  git clone https://github.com/limix/struct-lmm.git
  cd struct-lmm
  python setup.py install test

- install documentation::

  cd doc
  make html

