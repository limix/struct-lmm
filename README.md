# Struct-lmm

[![Travis](https://img.shields.io/travis/com/limix/struct-lmm.svg?style=flat-square&label=linux%20%2F%20macos%20build)](https://travis-ci.com/limix/struct-lmm) [![Documentation](https://img.shields.io/readthedocs/struct-lmm.svg?style=flat-square&version=stable)](https://struct-lmm.readthedocs.io/)

Structured Linear Mixed Model (StructLMM) is a computationally efficient method to test for and characterize loci that interact with multiple environments [1].

This a standalone module that implements the basic functionalities of StructLMM.
However, we recommend using StructLMM using [LIMIX2](https://limix.readthedocs.io/en/2.0.0/index.html) as this additionally implements:

- multiple methods for GWAS;
- methods to characterize GxE at specific variants;
- command line interface.

[1] Moore R, Casale FP, Bonder MJ, Horta D, Franke L, Barroso I, Stegle O, BIOS Consortium. A linear mixed model approach to study multivariate gene-environment interactions. bioRxiv. 2018 Jan 1:270611.

## Install

From terminal, it can be installed using [pip](https://pypi.python.org/pypi/pip):

```bash
pip install struct-lmm
```

## Documentation

The public interface and a quick start in python are available at
[http://struct-lmm.readthedocs.io/](http://struct-lmm.readthedocs.io/).

## Problems

If you encounter any problem, please, consider submitting a [new issue](https://github.com/limix/struct-lmm/issues/new).

## Authors

- **Francesco Paolo Casale** - [https://github.com/fpcasale](https://github.com/fpcasale)
- **Danilo Horta** - [https://github.com/horta](https://github.com/horta)
- **Rachel Moore** - [https://github.com/rm18](https://github.com/rm18)
- **Oliver Stegle** - [https://github.com/ostegle](https://github.com/ostegle)

## License

This project is licensed under the Apache License (Version 2.0, January 2004) -
see the [LICENSE](LICENSE) file for details
