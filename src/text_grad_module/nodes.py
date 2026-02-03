"""
LangGraph node wrappers for TextGrad module.
"""

from typing import TypedDict, Dict, Any, Optional
from .config import TextGradConfig


class TextGradState(TypedDict):
    """State structure for LangGraph integration."""
    text: str
    optimized_text: Optional[str]
    loss: Optional[float]
    iteration: int
    metadata: Dict[str, Any]
