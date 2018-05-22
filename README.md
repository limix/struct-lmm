# struct-lmm

Structured Linear Mixed Model (StructLMM) is a computationally efficient method
to test for and characterize loci that interact with multiple environments.

## Getting Started

StructLMM is Python package implemented in both Python and R programming
languages that depends on some libraries implemented in C.
To make its installation as easy as possible, we make use
[conda](https://conda.io/), a package manager for software implemented in
Python, R, and C/C++ (among others) languages.

Interaction with StructLMM happens in the terminal via the following command
line tools installed with the package, and described at the [Command Line Interface](http://struct-lmm.readthedocs.io/en/latest/commandline.html)
section of the [documentation](http://struct-lmm.readthedocs.io/):

- norm_env
- struct_lmm_analyze
- lmm_int_analyze
- lmm_analyze
- struct_lmm_analyze


### Requisites

The installation script runs in on GNU/Linux and macOS operating systems.
It requires either [wget](https://www.gnu.org/software/wget/) or
[curl](https://curl.haxx.se/) command line tools in case
[conda](https://conda.io/) is not already installed.
In any case, the installation script will inform the user if it cannot proceed.
Otherwise, the StructLMM dependencies is automatically installed and does not
require user intervention.

### Install

For Linux and macOS operating systems, struct-lmm can be install from the
command line by entering

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/limix/struct-lmm/master/install)
```

The user might be prompted to install conda in case he/she does not have
it, and will warn the user if for some reason the installation process cannot
proceed.
The whole installation process should take less than 15 minutes and mainly
consists in downloading essential R and Python packages for a working
environment.


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

* **Rachel Moore** - [https://github.com/rm18](https://github.com/rm18)
* **Danilo Horta** - [https://github.com/horta](https://github.com/horta)
* **Francesco Paolo Casale** - [https://github.com/fpcasale](https://github.com/fpcasale)


## License

This project is licensed under the Apache License (Version 2.0, January 2004) -
see the [LICENSE](LICENSE) file for details
