"""
TextGrad Module - A reusable module for TextGrad optimization and LangGraph integration.
"""

__version__ = "0.1.0"

from .config import TextGradConfig
from .optimizer import TextGradOptimizer
from .evaluator import TextGradEvaluator
from .nodes import (
    TextGradState,
    create_optimize_node,
    create_evaluate_node,
    create_compare_node,
    create_optimize_node_async,
    create_evaluate_node_async,
)

__all__ = [
    "TextGradConfig",
    "TextGradOptimizer",
    "TextGradEvaluator",
    "TextGradState",
    "create_optimize_node",
    "create_evaluate_node",
    "create_compare_node",
    "create_optimize_node_async",
    "create_evaluate_node_async",
]
