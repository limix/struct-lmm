# struct-lmm

[![Travis](https://img.shields.io/travis/com/limix/struct-lmm.svg?style=flat-square&label=linux%20%2F%20macos%20build)](https://travis-ci.com/limix/struct-lmm) [![Documentation](https://img.shields.io/readthedocs/struct-lmm.svg?style=flat-square&version=stable)](https://struct-lmm.readthedocs.io/)

Structured Linear Mixed Model (StructLMM) is a computationally efficient method
to test for and characterize loci that interact with multiple environments.

## Install

From terminal, it can be installed using [pip](https://pypi.python.org/pypi/pip):

```bash
pip install struct-lmm
```

## Getting Started

Interaction with StructLMM happens in the terminal via the following command
line tools installed with the package, and described at the [Command Line Interface](http://struct-lmm.readthedocs.io/en/latest/commandline.html)
section of the [documentation](http://struct-lmm.readthedocs.io/):

- norm_env
- struct_lmm_analyze
- lmm_int_analyze
- lmm_analyze
- struct_lmm_analyze

## Usage example

StructLMM can be run from the command line using the following

```bash
wget http://www.ebi.ac.uk/~casale/data_structlmm.zip
unzip data_structlmm.zip

BFILE=data_structlmm/chrom22_subsample20_maf0.10
PFILE=data_structlmm/expr.csv
EFILE0=data_structlmm/env.txt
EFILE=data_structlmm/env_norm.txt

norm_env  --in $EFILE0 --out $EFILE

struct_lmm_analyze --bfile $BFILE --pfile $PFILE --pheno_id gene10 --efile $EFILE --ofile out/results.res --idx_start 0 --idx_end 1000 --batch_size 100 --unique_variants
```

Further examples can be found at [http://struct-lmm.readthedocs.io/](http://struct-lmm.readthedocs.io/).

## Documentation

Documentation is available online at
[http://struct-lmm.readthedocs.io/](http://struct-lmm.readthedocs.io/).

## Problems

If you encounter any problem, please, consider submitting a [new issue](https://github.com/limix/struct-lmm/issues/new).

## Authors

- **Rachel Moore** - [https://github.com/rm18](https://github.com/rm18)
- **Danilo Horta** - [https://github.com/horta](https://github.com/horta)
- **Francesco Paolo Casale** - [https://github.com/fpcasale](https://github.com/fpcasale)

## License

This project is licensed under the Apache License (Version 2.0, January 2004) -
see the [LICENSE](LICENSE) file for details
