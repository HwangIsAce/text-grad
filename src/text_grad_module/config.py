"""
Configuration management for TextGrad module.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextGradConfig:
    """Configuration for TextGrad optimization."""
    
    backward_engine: str = field(default_factory=lambda: os.getenv("TEXTGRAD_BACKWARD_ENGINE", "gpt-4o"))
    forward_engine: Optional[str] = field(default_factory=lambda: os.getenv("TEXTGRAD_FORWARD_ENGINE", None))
    max_iterations: int = 1
    learning_rate: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
    
    @classmethod
    def from_env(cls) -> "TextGradConfig":
        """Create configuration from environment variables."""
        return cls(
            backward_engine=os.getenv("TEXTGRAD_BACKWARD_ENGINE", "gpt-4o"),
            forward_engine=os.getenv("TEXTGRAD_FORWARD_ENGINE", None),
            max_iterations=int(os.getenv("TEXTGRAD_MAX_ITERATIONS", "1")),
            learning_rate=float(os.getenv("TEXTGRAD_LEARNING_RATE", "1.0")),
        )
