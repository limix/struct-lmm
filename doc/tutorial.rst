*********************
Step by step tutorial
*********************

Here, we provide a step by step tutorial for performing a GxE analysis

We can integrate command line functions into the relevant sections below

1. Loading and prepocessing the data  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section we load in the different required data files and describe the expected data formats

Importing genotype data
-----------------------

Link here to bgen file formats?

Importing phenotype data
------------------------

Importing environmental data
----------------------------

Importing covariate data
------------------------

2. Preprocessing and visulisation of environmental data  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section we will demonstrate possible ways to preprocess the environmental data and how to visualise the sample covariance and correlation between the environments.

Environmental similarity visualisation
--------------------------------------
Correlation between environments

Standardising the environmental data
------------------------------------
Mean = 0, std = 1
We then recommend that you follow either the linear covariance stabilisation step or the correlation matrix step

Linear covariance stabilisation
-------------------------------
Division step

Linear covariance visualisation
-------------------------------
Visualisation

Correlation standardisation
---------------------------
row standardisation step

Correlation visualisation
-------------------------
Visualisation


3. Association tests 
^^^^^^^^^^^^^^^^^^^^
Here, we demonstrate how to perform StructLMM joint association test, LMM, joint fixed effect association tests with single or multiple environmental variables and how to visualise the results

StructLMM joint association test
--------------------------------

LMM association test with/without additive environmental covariance
-------------------------------------------------------------------

Fixed effect joint association test with single and multiple environmental variables
------------------------------------------------------------------------------------

QQ plots
--------

Manhattan plots
---------------



4. Interaction tests (StructLMM-int)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here, we demonstrate how to perform StructLMM interaction test, fixed effect interaction tests with single or multiple environmental variables and how to visualise the results

StructLMM interaction test
--------------------------

Fixed effect interaction test with single and multiple environmental variables
------------------------------------------------------------------------------

QQ plots
--------

Manhattan plots
---------------


6. Downstream interpretation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimate \rho
-------------

Predict per-individual genetic effects
--------------------------------------

Identify driving GxE environments
---------------------------------




First you have to run some command line stuff

.. code-block:: bash

    wget http://ichef.bbci.co.uk/wwfeatures/wm/live/1280_640/images/live/p0/52/1q/p0521q8t.jpg
    mv p0521q8t.jpg dali.jpg

Then run in python interpreter

.. code-block:: python

    import struct_lmm
    struct_lmm.run("dali.jpg")

.. image:: dali.jpg
   :width: 400px

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.show()
