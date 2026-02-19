"""
Baselines module for Phase 2 comparison experiments.

Provides:
  - BM25Retriever, LSARetriever: retrieval-based baselines
  - TemplateFiller: rule-based template filling baseline
  - ZeroShotGenerator: zero-shot and few-shot generation baseline
  - inference_utils: shared cache save/load and metric utilities
"""

from src.baselines.ir_methods import BM25Retriever, LSARetriever
from src.baselines.template_filling import TemplateFiller
from src.baselines.zero_shot import ZeroShotGenerator

__all__ = [
    'BM25Retriever',
    'LSARetriever',
    'TemplateFiller',
    'ZeroShotGenerator',
]