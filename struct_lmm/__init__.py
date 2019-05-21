"""
StructLMM
=========

Let n be the number of samples.
StructLMM [1] extends the conventional linear mixed model by including an
additional per-individual effect term that accounts for genotype-environment
interaction, which can be represented as an n×1 vector, 𝛃.
The model is given by

    𝐲 = 𝙼𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + 𝛆,

where

    𝛽 ∼ 𝓝(0, 𝓋₀⋅ρ), 𝛃 ∼ 𝓝(𝟎, 𝓋₀(1-ρ)𝙴𝙴ᵀ), 𝐞 ∼ 𝓝(𝟎, 𝓋₁𝚆𝚆ᵀ), and 𝛆 ∼ 𝓝(𝟎, 𝓋₂𝙸).

The vector 𝐲 is the outcome, matrix 𝙼 contains the covariates, and vector 𝐠 is the
genetic variant.
The matrices 𝙴 and 𝚆 are generally the same, and represent the environment
configuration for each sample.
The parameters 𝓋₀, 𝓋₁, and 𝓋₂ are the overall variances.
The parameter ρ ∈ [𝟶, 𝟷] dictates the relevance of genotype-environment interaction
versus the genotype effect alone.
The term 𝐞 accounts for additive environment-only effects while 𝛆 accounts for
noise effects.

.. [1] Moore, R., Casale, F. P., Bonder, M. J., Horta, D., Franke, L., Barroso, I., &
   Stegle, O. (2018). A linear mixed-model approach to study multivariate
   gene–environment interactions (p. 1). Nature Publishing Group.
"""
from ._lmm import StructLMM
from ._testit import test

__version__ = "0.3.1"

__all__ = ["StructLMM", "__version__", "test"]
