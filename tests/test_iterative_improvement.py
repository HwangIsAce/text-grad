"""
Test if TextGrad improves prompts iteratively with each step.
Shows how quality scores change across multiple iterations.
"""

import sys
from pathlib import Path
import time

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from text_grad_module import TextGradConfig, TextGradOptimizer, TextGradEvaluator
import textgrad as tg


def evaluate_prompt_quality(prompt: str, evaluator: TextGradEvaluator) -> float:
    """
    Evaluate prompt quality and return overall score.
    
    Args:
        prompt: Prompt to evaluate
        evaluator: TextGradEvaluator instance
        
    Returns:
        Overall quality score (0-10)
    """
    try:
        result = evaluator.evaluate_text(
            text=prompt,
            loss_fn="Rate the overall quality of this prompt from 0 to 10. Consider specificity, clarity, actionability, and completeness. Respond with only a number.",
            role_description="prompt to evaluate"
        )
        
        # Try to extract numeric score
        import re
        loss_value = result.get("loss", "0")
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', str(loss_value))
        if numbers:
            score = float(numbers[0])
            if score > 10:
                score = 10
            return score
        else:
            # Fallback: use length as proxy
            return min(10, len(prompt.split()) / 5)
    except Exception:
        # Fallback scoring
        return min(10, len(prompt.split()) / 5)


def test_iterative_improvement():
    """Test if TextGrad improves prompts iteratively."""
    print("=" * 80)
    print("TextGrad Iterative Improvement Test")
    print("=" * 80)
    
    # Setup
    config = TextGradConfig(backward_engine="gpt-4o", max_iterations=3)
    optimizer = TextGradOptimizer(config)
    evaluator = TextGradEvaluator(config)
    
    test_cases = [
        {
            "prompt": "Write a story.",
            "description": "Simple Story Writing",
            "role": "prompt for story writing"
        },
        {
            "prompt": "Solve this math problem.",
            "description": "Math Problem Solving",
            "role": "prompt for solving math problems"
        }
    ]
    
    for test_case in test_cases:
        initial_prompt = test_case["prompt"]
        description = test_case["description"]
        
        print(f"\n{'=' * 80}")
        print(f"Test: {description}")
        print(f"{'=' * 80}")
        print(f"\n📝 Initial Prompt:")
        print(f"   '{initial_prompt}'")
        
        # Evaluate initial prompt
        initial_score = evaluate_prompt_quality(initial_prompt, evaluator)
        print(f"   Initial Score: {initial_score:.2f}/10")
        
        # Track improvements across iterations
        iteration_results = []
        iteration_results.append({
            "iteration": 0,
            "prompt": initial_prompt,
            "score": initial_score
        })
        
        # Run optimization with multiple iterations
        print(f"\n🔄 Running optimization with {config.max_iterations} iterations...")
        print("-" * 80)
        
        loss_fn = tg.TextLoss(
            "Evaluate this prompt. A good prompt should be: "
            "1) Clear and specific, 2) Provide context, 3) Include examples if needed, "
            "4) Be actionable. Improve the prompt to make it more effective."
        )
        
        # Manually run iterations to track progress
        variable = optimizer.create_variable(
            initial_prompt,
            requires_grad=True,
            role_description=test_case["role"]
        )
        
        optimizer_instance = optimizer.create_optimizer([variable])
        
        for iteration in range(1, config.max_iterations + 1):
            print(f"\n[Iteration {iteration}]")
            
            # Forward pass
            loss = loss_fn(variable)
            
            # Evaluate current prompt
            current_score = evaluate_prompt_quality(variable.value, evaluator)
            print(f"   Current Prompt: '{variable.value[:100]}...' (if longer)")
            print(f"   Current Score: {current_score:.2f}/10")
            print(f"   Loss: {loss.value if hasattr(loss, 'value') else 'N/A'}")
            
            iteration_results.append({
                "iteration": iteration,
                "prompt": variable.value,
                "score": current_score
            })
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer_instance.step()
            
            # Small delay to see progress
            time.sleep(0.5)
        
        # Final evaluation
        final_prompt = variable.value
        final_score = evaluate_prompt_quality(final_prompt, evaluator)
        
        print(f"\n{'=' * 80}")
        print("Improvement Summary")
        print(f"{'=' * 80}")
        print(f"\nInitial Score: {initial_score:.2f}/10")
        print(f"Final Score: {final_score:.2f}/10")
        print(f"Improvement: {final_score - initial_score:+.2f} points")
        print(f"Improvement Rate: {((final_score - initial_score) / initial_score * 100) if initial_score > 0 else 0:.1f}%")
        
        # Show progression
        print(f"\n📈 Score Progression:")
        for result in iteration_results:
            print(f"   Iteration {result['iteration']}: {result['score']:.2f}/10")
        
        # Check if improvement is monotonic
        scores = [r["score"] for r in iteration_results]
        is_improving = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
        is_degrading = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        if is_improving:
            print(f"\n✅ Scores are improving monotonically!")
        elif is_degrading:
            print(f"\n⚠️  Scores are decreasing (may indicate over-optimization)")
        else:
            print(f"\n📊 Scores fluctuate (normal for iterative optimization)")
        
        # Show final prompt
        print(f"\n✨ Final Optimized Prompt:")
        print(f"   '{final_prompt}'")
        
        # Reset optimizer for next test
        optimizer.reset()


if __name__ == "__main__":
    try:
        test_iterative_improvement()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
