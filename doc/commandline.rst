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

* Run analysis with struct LMM::

    python struct_lmm_analyze.py --bfile $BFILE \
                                 --pfile $PFILE \
                                 --pheno_id gene10 \
                                 --efile $EFILE \
                                 --ofile out/results.res \
                                 --idx_start 0 \
                                 --idx_end 1000 \
                                 --batch_size 100 \
                                 --unique_variants

* Run analysis with standard LMM::

    python lmm_lr.py --bfile $BFILE \
                     --pfile $PFILE \
                     --pheno_id gene10 \
                     --wfile $EFILE \
                     --ofile out/results.res \
                     --idx_start 0 \
                     --idx_end 1000 \
                     --batch_size 100 \
                     --unique_variants



Commands 
~~~~~~~~

**struct_lmm_analyze**
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  python struct_lmm_analyze.py --bfile bfile \
                               --pfile pfile \
                               --pheno_id pheno_id \
                               --efile efile \
                               --ffile ffile \
                               --ofile ofile \
                               --idx_start idx_start \
                               --idx_end idx_end \
                               --pos_start pos_start \
                               --pos_end pos_end \
                               - chrom pos_chrom \
                               --batch_size bathc_size \
                               --unique_variants \
                               --no_interaction_test \
                               --no_mean_to_one


* **bfile** is the base name of of the binary bed file
* **pfile** phenotype file name (see :ref:`formats_ref`) 
* **pheno_id** phenotype identifier as in the phenotype file 
* **efile** environment file name (see :ref:`formats_ref`) 
* **ffile** covariate file name (see :ref:`formats_ref`).
  If not set, only an intercept is considered
* **ofile** output file. It contains pvalues for both the joint
  and the intaction test as well as snp info.
  If **no_interaction_test** is specified the interaction test
  is not considered.
* **idx_start** idx of snp for which the analyses should start.
  If specfied, the query 'idx >= idx_start' on genotype data is applied.
* **idx_end** idx of snp for which the aalysis should end.
  If specified, the query 'idx < idx_end' on genotype data is applied.
* **pos_start** start position (genotype query).
* **pos_end** end position (genotype query).
* **chrom** chrom (genotype query).
* **batch_size**. To minimize memory usage the analysis is run in batches.
  The number of variants loaded in a batch (in memory at the same time).
* **no_interaction_test**. If active the interaction test is not consdered.
* **unique_variants**. If activated, only non-repeated genotypes are considered.
* **no_mean_to_one**. 
  when not activated, the environment matrix is normalized in such
  a way that the outer product EE^T has diagonal of ones.
  if activated, the environment matrix is normalized in such
  a way that the outer product EE^T has mean of diagonal
  of ones.

**lmm_lr**
^^^^^^^^^^

.. code-block:: bash

  python struct_lmm_analyze.py --bfile bfile \
                               --pfile pfile \
                               --pheno_id pheno_id \
                               --wfile wfile \
                               --ffile ffile \
                               --ofile ofile \
                               --idx_start idx_start \
                               --idx_end idx_end \
                               --pos_start pos_start \
                               --pos_end pos_end \
                               - chrom pos_chrom \
                               --batch_size bathc_size \
                               --unique_variants \
                               --no_mean_to_one


* **wfile** file that defines the low rank random effect (see :ref:`formats_ref`).

See above for other parameters.

.. _formats_ref:

Formats
~~~~~~~

* **bfile** are plink bed
* **pfile** (phenotype file) is assumed to be a csv file with dimension #pheno by #individuals and
  having row and col readers.
  See example at http://www.ebi.ac.uk/~casale/example_data/expr.csv.
* **efile** (environment file) is a tsv file with dimensions #inds by #environments.
  See example at http://www.ebi.ac.uk/~casale/data_structlmm/env.txt
* **ffile** (covariates file) is a tsv file with dimensions #inds by #covariates.
  Should contain a column of ones to include an intercept in the model.
* **wfile** (random eff design file) is a tsv file with dimensions #inds by #random effects that defines the random effect.

