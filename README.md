# Struct-LMM

[![Travis](https://img.shields.io/travis/com/limix/struct-lmm.svg?style=flat-square&label=linux%20%2F%20macos%20build)](https://travis-ci.com/limix/struct-lmm) [![AppVeyor](https://img.shields.io/appveyor/ci/Horta/struct-lmm-rxwgm.svg?style=flat-square&label=windows%20build)](https://ci.appveyor.com/project/Horta/struct-lmm-rxwgm)

Structured Linear Mixed Model (StructLMM) is a computationally efficient method to
test for and characterize loci that interact with multiple environments [1].

This a standalone module that implements the basic functionalities of StructLMM.
However, we recommend using StructLMM via
[LIMIX2](https://limix.readthedocs.io/en/2.0.x/index.html) as this additionally
implements:

- Multiple methods for GWAS;
- Methods to characterize GxE at specific variants;
- Command line interface.

## Install

From terminal, it can be installed using [pip](https://pypi.org/pypi/pip):

```bash
pip install struct-lmm
```

## Usage

```python
>>> from numpy import ones, concatenate
>>> from numpy.random import RandomState
>>>
>>> from struct_lmm import StructLMM
>>>
>>> random = RandomState(1)
>>> n = 20
>>> k = 4
>>> y = random.randn(n, 1)
>>> E = random.randn(n, k)
>>> M = ones((n, 1))
>>> x = 1.0 * (random.rand(n, 1) < 0.2)
>>>
>>> lmm = StructLMM(y, M, E)
>>> lmm.fit(verbose=False)
>>> pv = lmm.score_2dof_assoc(x)
>>> print(pv)
0.8470017194859742
>>> M = concatenate([M, x], axis=1)
>>> lmm = StructLMM(y, M, E)
>>> lmm.fit(verbose=False)
>>> pv = lmm.score_2dof_inter(x)
>>> print(pv)
0.6781100453132024
```

## Problems

If you encounter any problem, please, consider submitting a [new issue](https://github.com/limix/struct-lmm/issues/new).

## Authors

- [Danilo Horta](https://github.com/horta)
- [Francesco Paolo Casale](https://github.com/fpcasale)
- [Oliver Stegle](https://github.com/ostegle)
- [Rachel Moore](https://github.com/rm18)

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/limix/struct-lmm/master/LICENSE.md).

[1] Moore, R., Casale, F. P., Bonder, M. J., Horta, D., Franke, L., Barroso, I., &
    Stegle, O. (2018). [A linear mixed-model approach to study multivariate
    gene–environment interactions](https://www.nature.com/articles/s41588-018-0271-0) (p. 1). Nature Publishing Group.
