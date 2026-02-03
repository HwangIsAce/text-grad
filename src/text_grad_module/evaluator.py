"""
TextGrad evaluator module.
"""

import textgrad as tg
from typing import Optional, Callable, Dict, Any, Union
from .config import TextGradConfig


class TextGradEvaluator:
    """Evaluator for TextGrad optimization results."""
    
    def __init__(self, config: Optional[TextGradConfig] = None):
        """
        Initialize TextGrad evaluator.
        
        Args:
            config: Configuration for TextGrad. If None, uses default config.
        """
        self.config = config or TextGradConfig()
    
    def create_loss_fn(self, loss_prompt: str) -> tg.TextLoss:
        """
        Create a TextGrad loss function.
        
        Args:
            loss_prompt: Prompt for the loss function
            
        Returns:
            TextGrad TextLoss object
        """
        return tg.TextLoss(loss_prompt)
    
    def evaluate_text(
        self,
        text: str,
        loss_fn: Union[tg.TextLoss, str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate text using a loss function.
        
        Args:
            text: Text to evaluate
            loss_fn: Loss function or prompt string
            ground_truth: Optional ground truth for comparison
            
        Returns:
            Dictionary with evaluation results:
            - loss: Loss value
            - text: Evaluated text
            - ground_truth: Ground truth (if provided)
        """
        if isinstance(loss_fn, str):
            loss_fn = self.create_loss_fn(loss_fn)
        
        # Create variable for evaluation
        variable = tg.Variable(text, requires_grad=False, role_description="text to evaluate")
        
        # Compute loss
        loss = loss_fn(variable)
        
        result = {
            "loss": loss.value if hasattr(loss, 'value') else str(loss),
            "text": text,
        }
        
        if ground_truth:
            result["ground_truth"] = ground_truth
            # Simple comparison (can be enhanced)
            result["matches"] = text.strip().lower() == ground_truth.strip().lower()
        
        return result
    
    def compare_texts(
        self,
        original: str,
        optimized: str,
        loss_fn: Union[tg.TextLoss, str]
    ) -> Dict[str, Any]:
        """
        Compare original and optimized texts.
        
        Args:
            original: Original text
            optimized: Optimized text
            loss_fn: Loss function or prompt string
            
        Returns:
            Dictionary with comparison results:
            - original_loss: Loss of original text
            - optimized_loss: Loss of optimized text
            - improvement: Whether optimization improved the text
            - improvement_ratio: Ratio of improvement
        """
        original_result = self.evaluate_text(original, loss_fn)
        optimized_result = self.evaluate_text(optimized, loss_fn)
        
        # Try to extract numeric loss values for comparison
        try:
            orig_loss = float(original_result["loss"])
            opt_loss = float(optimized_result["loss"])
            improvement = opt_loss < orig_loss
            improvement_ratio = (orig_loss - opt_loss) / orig_loss if orig_loss > 0 else 0
        except (ValueError, TypeError):
            # If loss is not numeric, use string comparison
            improvement = str(optimized_result["loss"]) < str(original_result["loss"])
            improvement_ratio = None
        
        return {
            "original_loss": original_result["loss"],
            "optimized_loss": optimized_result["loss"],
            "improvement": improvement,
            "improvement_ratio": improvement_ratio,
            "original_text": original,
            "optimized_text": optimized,
        }
