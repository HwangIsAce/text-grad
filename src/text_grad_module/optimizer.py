"""
TextGrad optimizer module.
"""

import textgrad as tg
from typing import Optional, List, Dict, Any
from .config import TextGradConfig


class TextGradOptimizer:
    """Wrapper for TextGrad optimization functionality."""
    
    def __init__(self, config: Optional[TextGradConfig] = None):
        """
        Initialize TextGrad optimizer.
        
        Args:
            config: Configuration for TextGrad. If None, uses default config.
        """
        self.config = config or TextGradConfig()
        self._setup_engines()
        self._variables: List[tg.Variable] = []
        self._optimizer: Optional[tg.TGD] = None
    
    def _setup_engines(self):
        """Set up TextGrad engines."""
        try:
            tg.set_backward_engine(self.config.backward_engine, override=True)
        except TypeError:
            # If override parameter doesn't exist, try without it
            tg.set_backward_engine(self.config.backward_engine)
        if self.config.forward_engine:
            # Note: TextGrad may not have set_forward_engine, so we handle it gracefully
            try:
                tg.set_forward_engine(self.config.forward_engine)
            except AttributeError:
                pass
    
    def create_variable(
        self,
        value: str,
        requires_grad: bool = True,
        role_description: Optional[str] = None
    ) -> tg.Variable:
        """
        Create a TextGrad Variable.
        
        Args:
            value: Initial text value
            requires_grad: Whether to enable gradient computation
            role_description: Description of the variable's role
            
        Returns:
            TextGrad Variable object
        """
        variable = tg.Variable(
            value,
            requires_grad=requires_grad,
            role_description=role_description
        )
        self._variables.append(variable)
        return variable
    
    def create_optimizer(self, parameters: Optional[List[tg.Variable]] = None) -> tg.TGD:
        """
        Create a TextGrad optimizer.
        
        Args:
            parameters: List of variables to optimize. If None, uses all tracked variables.
            
        Returns:
            TextGrad TGD optimizer
        """
        if parameters is None:
            parameters = [v for v in self._variables if v.requires_grad]
        
        self._optimizer = tg.TGD(parameters=parameters)
        return self._optimizer
    
    def optimize(
        self,
        initial_text: str,
        loss_fn: tg.TextLoss,
        role_description: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize text using TextGrad.
        
        Args:
            initial_text: Initial text to optimize
            loss_fn: TextGrad loss function
            role_description: Description of the text variable's role
            max_iterations: Maximum optimization iterations. Uses config value if None.
            
        Returns:
            Dictionary with optimization results:
            - optimized_text: Optimized text
            - loss: Final loss value
            - iterations: Number of iterations performed
            - history: List of (iteration, text, loss) tuples
        """
        iterations = max_iterations or self.config.max_iterations
        history = []
        
        # Create variable
        variable = self.create_variable(
            initial_text,
            requires_grad=True,
            role_description=role_description
        )
        
        # Create optimizer
        optimizer = self.create_optimizer([variable])
        
        # Optimization loop
        for iteration in range(iterations):
            # Forward pass
            loss = loss_fn(variable)
            
            # Record history
            history.append({
                "iteration": iteration,
                "text": variable.value,
                "loss": loss.value if hasattr(loss, 'value') else str(loss)
            })
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
        
        return {
            "optimized_text": variable.value,
            "loss": loss.value if hasattr(loss, 'value') else None,
            "iterations": iterations,
            "history": history
        }
    
    def reset(self):
        """Reset optimizer state."""
        self._variables = []
        self._optimizer = None
