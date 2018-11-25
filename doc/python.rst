.. _python:

***********
Quick Start
***********

.. Attention::

    We here show how to run the stand alone version of StructLMM in Python.
    However, we recommend using StructLMM whithin `LIMIX2` as it additionally implements:

    - multiple `methods for GWAS`_;
    - `methods to characterize GxE`_ at specific variants;
    - `command line interface`_.

Let's get some data::

    wget http://www.ebi.ac.uk/~casale/data_structlmm.zip
    unzip data_structlmm.zip

Now we are ready to go.

.. literalinclude:: example.py
   :encoding: latin-1

The following script can be downloader :download:`here <example.py>`.

The core function to run structlmm is

- :func:`.struct_lmm.runner.run_struct_lmm`

and are described in :ref:`public`.


.. _LIMIX2: https://github.com/limix/limix/tree/2.0.0/limix
.. _`methods for GWAS`: https://limix.readthedocs.io/en/2.0.0/gwas.html
.. _`command line interface`: https://limix.readthedocs.io/en/2.0.0/cmd.html
.. _`methods to characterize GxE`: https://limix.readthedocs.io/en/2.0.0
