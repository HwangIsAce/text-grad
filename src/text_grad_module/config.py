"""
Configuration management for TextGrad module.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TextGradConfig:
    """Configuration for TextGrad optimization."""
    
    backward_engine: str = "gpt-4o"
    forward_engine: Optional[str] = None
    max_iterations: int = 1
    learning_rate: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
