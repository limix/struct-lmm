# struct-lmm

Describe me

## Install

The package depends scipy, numpy, rpy2, limix-core.

- Create a new environment in [conda](https://conda.io/docs/index.html)
  ```bash
  conda create -n struct-lmm python=2.7
  source activate struct-lmm
  ```

- install numpy, scipy, rpy2 and limix-core
  ```
  conda install -n struct-lmm numpy scipy
  conda install -c r -n struct-lmm rpy2
  pip install limix-core
  ```

- install struct-lmm (hopefully all tests pass)
  ```bash
  git clone https://github.com/limix/struct-lmm.git
  cd struct-lmm
  python setup.py install test
  ```

- install documentation
  ```bash
  cd doc
  make html
  ```
  

## Documentation

Documentation is available in struct-lmm/doc/html/index.html.

## Problems

If you encounter any issue, please [submit it](https://github.com/limix/struct-lmm/issues).

## Authors

* **Francesco Paolo Casale** - [https://github.com/fpcasale](https://github.com/fpcasale)
* **Rachel Moore** - [https://github.com/rm18](https://github.com/rm18)

## License

This project is licensed under the Apache License (Version 2.0, January 2004) -
see the [LICENSE](LICENSE) file for details
