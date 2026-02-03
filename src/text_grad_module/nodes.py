"""
LangGraph node wrappers for TextGrad module.
"""

from typing import TypedDict, Dict, Any, Optional, Callable
import textgrad as tg
from .config import TextGradConfig
from .optimizer import TextGradOptimizer
from .evaluator import TextGradEvaluator


class TextGradState(TypedDict):
    """State structure for LangGraph integration."""
    text: str
    optimized_text: Optional[str]
    loss: Optional[float]
    iteration: int
    metadata: Dict[str, Any]


def create_optimize_node(
    config: Optional[TextGradConfig] = None,
    loss_prompt: Optional[str] = None
) -> Callable[[TextGradState], TextGradState]:
    """
    Create a LangGraph node for text optimization.
    
    Args:
        config: TextGrad configuration
        loss_prompt: Loss function prompt. If None, must be provided in state.
        
    Returns:
        Node function that takes state and returns updated state
    """
    optimizer = TextGradOptimizer(config or TextGradConfig())
    
    def optimize_node(state: TextGradState) -> TextGradState:
        """Optimize text node."""
        text = state.get("text", "")
        if not text:
            return state
        
        # Get loss prompt from state or use provided one
        prompt = state.get("metadata", {}).get("loss_prompt") or loss_prompt
        if not prompt:
            raise ValueError("loss_prompt must be provided in state metadata or as parameter")
        
        loss_fn = tg.TextLoss(prompt)
        role_description = state.get("metadata", {}).get("role_description")
        
        # Optimize
        result = optimizer.optimize(
            initial_text=text,
            loss_fn=loss_fn,
            role_description=role_description,
            max_iterations=state.get("metadata", {}).get("max_iterations")
        )
        
        # Update state
        new_state = state.copy()
        new_state["optimized_text"] = result["optimized_text"]
        new_state["loss"] = result.get("loss")
        new_state["iteration"] = state.get("iteration", 0) + 1
        new_state["metadata"] = state.get("metadata", {}).copy()
        new_state["metadata"]["optimization_history"] = result["history"]
        
        return new_state
    
    return optimize_node


def create_evaluate_node(
    config: Optional[TextGradConfig] = None
) -> Callable[[TextGradState], TextGradState]:
    """
    Create a LangGraph node for text evaluation.
    
    Args:
        config: TextGrad configuration
        
    Returns:
        Node function that takes state and returns updated state
    """
    evaluator = TextGradEvaluator(config or TextGradConfig())
    
    def evaluate_node(state: TextGradState) -> TextGradState:
        """Evaluate text node."""
        text = state.get("optimized_text") or state.get("text", "")
        if not text:
            return state
        
        # Get loss prompt from metadata
        loss_prompt = state.get("metadata", {}).get("loss_prompt")
        if not loss_prompt:
            raise ValueError("loss_prompt must be provided in state metadata")
        
        ground_truth = state.get("metadata", {}).get("ground_truth")
        
        # Evaluate
        result = evaluator.evaluate_text(
            text=text,
            loss_fn=loss_prompt,
            ground_truth=ground_truth
        )
        
        # Update state
        new_state = state.copy()
        new_state["loss"] = result.get("loss")
        new_state["metadata"] = state.get("metadata", {}).copy()
        new_state["metadata"]["evaluation"] = result
        
        return new_state
    
    return evaluate_node


def create_compare_node(
    config: Optional[TextGradConfig] = None
) -> Callable[[TextGradState], TextGradState]:
    """
    Create a LangGraph node for comparing original and optimized texts.
    
    Args:
        config: TextGrad configuration
        
    Returns:
        Node function that takes state and returns updated state
    """
    evaluator = TextGradEvaluator(config or TextGradConfig())
    
    def compare_node(state: TextGradState) -> TextGradState:
        """Compare texts node."""
        original = state.get("text", "")
        optimized = state.get("optimized_text", "")
        
        if not original or not optimized:
            return state
        
        # Get loss prompt from metadata
        loss_prompt = state.get("metadata", {}).get("loss_prompt")
        if not loss_prompt:
            raise ValueError("loss_prompt must be provided in state metadata")
        
        # Compare
        result = evaluator.compare_texts(
            original=original,
            optimized=optimized,
            loss_fn=loss_prompt
        )
        
        # Update state
        new_state = state.copy()
        new_state["metadata"] = state.get("metadata", {}).copy()
        new_state["metadata"]["comparison"] = result
        
        return new_state
    
    return compare_node


# Async versions for LangGraph async support
async def create_optimize_node_async(
    config: Optional[TextGradConfig] = None,
    loss_prompt: Optional[str] = None
) -> Callable[[TextGradState], TextGradState]:
    """Async version of optimize node."""
    sync_node = create_optimize_node(config, loss_prompt)
    
    async def async_optimize_node(state: TextGradState) -> TextGradState:
        """Async optimize text node."""
        return sync_node(state)
    
    return async_optimize_node


async def create_evaluate_node_async(
    config: Optional[TextGradConfig] = None
) -> Callable[[TextGradState], TextGradState]:
    """Async version of evaluate node."""
    sync_node = create_evaluate_node(config)
    
    async def async_evaluate_node(state: TextGradState) -> TextGradState:
        """Async evaluate text node."""
        return sync_node(state)
    
    return async_evaluate_node
