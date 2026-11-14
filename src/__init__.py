"""
ladp/src — Loss-Aware Dynamic Projection (LADP)
================================================
Public API for the LADP pipeline.

Typical usage
-------------
::

    from ladp.src import (
        LlavaAdapter,
        LossEstimator, ReconstructionNet, ReconstructionTrainer, NormStats,
        CLIPRelevanceScorer, LLMEmbeddingRelevanceScorer,
        Normaliser,
        PriorityModule,
    )
"""

from .connector_adapter import LlavaAdapter
from .loss_estimator import ReconstructionNet, LossEstimator, ReconstructionTrainer, NormStats
from .relevance_scorer import CLIPRelevanceScorer, LLMEmbeddingRelevanceScorer
from .normaliser import Normaliser
from .priority_module import PriorityModule

__all__ = [
    "LlavaAdapter",
    "ReconstructionNet", "LossEstimator", "ReconstructionTrainer", "NormStats",
    "CLIPRelevanceScorer", "LLMEmbeddingRelevanceScorer",
    "Normaliser",
    "PriorityModule",
]
