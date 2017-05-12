.. _python:

*********************
Quick Start in Python
*********************

We here show how to run structLMM and alternative linear
mixed models implementations in Python.

The command line interface is shocased at :ref:`commandline`.

Before getting started, let's get some data::

    wget http://www.ebi.ac.uk/~casale/data_structlmm.zip
    unzip data_structlmm.zip

Now we are ready to go.

.. literalinclude:: example.py
   :encoding: latin-1

The following script can be downloader :download:`here <example.py>`.

The core functions to run the different lmms are

- :func:`structLMM.run_struct_lmm`

and are described in :ref:`public`.

Other important functions and classes used here are:

- :func:`structLMM.utils.import_one_pheno_from_csv`
- :func:`structLMM.utils.make_out_dir`
- :class:`limix.data.BedReader`
- :func:`limix.data.build_geno_query`
- :func:`limix.data.util.unique_variants`


