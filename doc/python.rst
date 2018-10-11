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

The core function to run structlmm is 

- :func:`.struct_lmm.runner.run_struct_lmm`

and are described in :ref:`public`.

