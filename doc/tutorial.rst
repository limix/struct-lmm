*********************
Step by step tutorial
*********************

Here, we provide a step by step tutorial for performing a GxE analysis.  This tutorial requires sample data, which can be retrieved from here... using the following commands

We can integrate command line functions into the relevant sections below

1. Loading and prepocessing the data  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section we load in the different required data files and describe the expected data formats

Importing genotype data
-----------------------

If the data is in bed file format, could just suggest converting to bgen format and then work from there?

Link here to bgen file formats?
Currently everything is set up in the context of bed files but I think the ability to import bgen files along with various preprocessing functions is important?

Should we also provide/describe here some useful bgen QC functions and demonstrate?


Importing phenotype data
------------------------

In python interpreter

.. code-block:: python

    pheno_file = './data/pheno.txt'
    # Not sure that we want to do it this way - we can just read in a text file?
    y = import_one_pheno_from_csv(phenofile, pheno_id='gene1', standardize=True)

The phenotype data should be an N by 1 array.

We can visualise the phenotype distribution
.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    plt.hist(y, bins = 20)
    plt.xlabel('Frequency')
    plt.ylabel('Phenotype')
    plt.show()    

We can transform the phenotype data, for example we may wish to use the inverse normal transformation

.. code-block:: python

    y = 

We can again visualise the phenotype distribution and see that it is now normally distributed
.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    plt.hist(y, bins = 20)
    plt.xlabel('Frequency')
    plt.ylabel('Phenotype')
    plt.show()    

Importing environmental data
----------------------------

The environmental data can be imported as follows.

The environmental data should be an N by E array.

We will demonstrate ways of standardising the data here (need to work out how to hyperlink).

Importing covariate data
------------------------

Import covariates such as age, gender, batch.

In python interpreter

.. code-block:: python

    covariate_file = './data/covs.txt'
    covs = sp.loadtxt(covariate_file)

The covariate data is an N by 11 array

Now we also want to include 10 genetic principal components within our model. We will load and add these to the covariate matrix

.. code-block:: python

    principle_components_file = './data/pcs.txt'
    pcs = sp.loadtxt(principle_components_file)
    covs = sp.hstack((covs, pcs))

It is also important that the covariates contains a column of ones (this is the intercept of the model)
.. code-block:: python

    ones = sp.ones((covs.shape[0], 1))
    covs = sp.hstack((ones, covs))

We suggest that you standardise the covariates (apart from the intercept) for stability reasons.  This can be done as follows
.. code-block:: python

    covs[:, 1:] -= covs[:, 1:].mean(0)
    covs[:, 1:] /= covs[:, 1:].std(0)


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
