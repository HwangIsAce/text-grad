"""
Basic usage examples for TextGrad module.
"""

import textgrad as tg
from text_grad_module import TextGradConfig, TextGradOptimizer, TextGradEvaluator


def example_basic_optimization():
    """Example: Basic text optimization."""
    print("=== Basic Text Optimization ===")
    
    # Configure TextGrad
    config = TextGradConfig(
        backward_engine="gpt-4o",
        max_iterations=1
    )
    
    # Create optimizer
    optimizer = TextGradOptimizer(config)
    
    # Initial text to optimize
    initial_text = "This is a test sentence that needs improvement."
    
    # Create loss function
    loss_fn = tg.TextLoss(
        "Evaluate if this text is clear, concise, and well-written. "
        "Identify any issues and suggest improvements."
    )
    
    # Optimize
    result = optimizer.optimize(
        initial_text=initial_text,
        loss_fn=loss_fn,
        role_description="text to be optimized"
    )
    
    print(f"Original: {initial_text}")
    print(f"Optimized: {result['optimized_text']}")
    print(f"Iterations: {result['iterations']}")
    print()


def example_text_evaluation():
    """Example: Text evaluation."""
    print("=== Text Evaluation ===")
    
    # Create evaluator
    evaluator = TextGradEvaluator()
    
    # Text to evaluate
    text = "This is a well-written and clear sentence."
    
    # Evaluate
    result = evaluator.evaluate_text(
        text=text,
        loss_fn="Rate the quality of this text from 0 to 10.",
        ground_truth="This is a well-written and clear sentence."
    )
    
    print(f"Text: {text}")
    print(f"Loss: {result['loss']}")
    print(f"Matches ground truth: {result.get('matches', False)}")
    print()


def example_text_comparison():
    """Example: Compare original and optimized texts."""
    print("=== Text Comparison ===")
    
    evaluator = TextGradEvaluator()
    
    original = "Bad text."
    optimized = "This is a much better and more detailed text with clear meaning."
    
    result = evaluator.compare_texts(
        original=original,
        optimized=optimized,
        loss_fn="Rate the quality of this text from 0 to 10."
    )
    
    print(f"Original: {original}")
    print(f"Optimized: {optimized}")
    print(f"Original Loss: {result['original_loss']}")
    print(f"Optimized Loss: {result['optimized_loss']}")
    print(f"Improved: {result['improvement']}")
    print()


def example_config_from_env():
    """Example: Using configuration from environment variables."""
    print("=== Configuration from Environment ===")
    
    # Set environment variables before running:
    # export TEXTGRAD_BACKWARD_ENGINE="gpt-3.5-turbo"
    # export TEXTGRAD_MAX_ITERATIONS="2"
    
    config = TextGradConfig.from_env()
    print(f"Backward Engine: {config.backward_engine}")
    print(f"Max Iterations: {config.max_iterations}")
    print(f"Learning Rate: {config.learning_rate}")
    print()


if __name__ == "__main__":
    print("TextGrad Module - Usage Examples\n")
    
    # Uncomment the examples you want to run
    # Note: These require API keys and may incur costs
    
    # example_basic_optimization()
    # example_text_evaluation()
    # example_text_comparison()
    # example_config_from_env()
    
    print("Uncomment examples in the code to run them.")
    print("Make sure to set your API keys before running.")
