.. _commandline:

**********************
Command Line Interface 
**********************

StructLMM can also be run from command line as shown below.

Quick example
~~~~~~~~~~~~~

* Download sample data from http://www.ebi.ac.uk/~casale/data_structlmm.zip::

    wget http://www.ebi.ac.uk/~casale/data_structlmm.zip
    unzip data_structlmm.zip

* Define some variable::

    BFILE=data_structlmm/chrom22_subsample20_maf0.10
    PFILE=data_structlmm/expr.csv
    EFILE=data_structlmm/env.txt

* Run analysis::

    python struct_lmm_analyze.py --bfile $BFILE --pfile $PFILE --pheno_id gene10 --efile $EFILE --ofile out/results.res --idx_start 0 --idx_end 1000 --batch_size 100 --unique_variants


Formats
~~~~~~~

* Bed file
* Pheno file
* Env file

Command ``struct_lmm_analyze``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

python struct_lmm_analyze.py --bfile $BFILE --pfile $PFILE --pheno_id gene10 --efile $EFILE --ofile out/results.res --idx_start 0 --idx_end 1000 --batch_size 100 --unique_variants --no_interaction_test

where
* __bfile__ is the base name of of the binary bed file (__bfile__.bim is required).
* __window\_size__ is the size of the window (in basepairs). The default value is 30kb.
* __wfile__ is the base name of the output file.
  If not specified, the file is saved as __bfile__.window\_size.wnd in the current folder (output format described above).
* __plot\_windows__ if the flag is set, a histogram over the number of markers within a window is generated and saved as __wfile__.pdf.

- __start\_wnd__ is the index of the start window
- __end\_wnd__ is the index of the end window
- __minSnps__ if set only windows containing at least minSnps are considered in the analysis
rdir is the directory to which the results are exported.
- __n_perms__ number of null (sampled) test statistics (obtained thrugh permutations/parametric bootstraps)
- __rdir__ is the directory to which the results are exported. The command exports files *start_wnd*_*end_wnd*.iSet.real that contains test statistics and vairance components and *start_wnd*_*end_wnd*.iSet.perm that contains null statistics
- __ifile__ is the file path to a csv file containing an indicator (True or False) for each sample. If specified the analysis is performed for a stratified design.
- __startwnd\_endwnd__.res and contains results in the following format: window index, chromosome, start position, stop position, index of startposition, number of SNPs and log likelihood ratio.
