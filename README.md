# struct-lmm

The structured linear mixed model (StructLMM) is a computationally efficient method to test for and characterize loci that interact with multiple environments.

See link for more details.

## Install

- Create a new environment in [conda](https://conda.io/docs/index.html)
  ```bash
  conda create -n struct-lmm python=2.7
  source activate struct-lmm
  ```

- install dependencies
  ```
  conda install -c conda-forge limix r r-base r-essentials rpy2 r-compquadform sphinx sphinx_rtd_theme
  ```

- install struct-lmm (hopefully all tests pass)
  ```bash
  git clone https://github.com/limix/struct-lmm.git
  cd struct-lmm
  python setup.py install
  ```

- install documentation
  ```bash
  cd doc
  make html
  open _build/html/index.html
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
