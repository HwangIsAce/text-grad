"""
Test script to verify TextGrad prompt optimization effectiveness.
Shows how prompts are improved through optimization.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from text_grad_module import TextGradConfig, TextGradOptimizer, TextGradEvaluator
import textgrad as tg

def test_prompt_optimization_effectiveness():
    """Test if TextGrad actually improves prompts."""
    print("=" * 70)
    print("TextGrad Prompt Optimization Effectiveness Test")
    print("=" * 70)
    
    # Setup
    config = TextGradConfig(backward_engine="gpt-4o", max_iterations=1)
    optimizer = TextGradOptimizer(config)
    evaluator = TextGradEvaluator(config)
    
    # Test 1: Simple prompt optimization
    print("\n[Test 1] Simple Prompt Optimization")
    print("-" * 70)
    
    initial_prompt = "Write a story."
    loss_fn = tg.TextLoss(
        "Evaluate this prompt. A good prompt should be: "
        "1) Clear and specific, 2) Provide context, 3) Include examples if needed, "
        "4) Be actionable. Rate the prompt quality from 0-10 where 0 is poor and 10 is excellent."
    )
    
    # Evaluate original
    print(f"\n📝 Original Prompt:")
    print(f"   '{initial_prompt}'")
    original_eval = evaluator.evaluate_text(
        initial_prompt,
        loss_fn="Rate this prompt quality from 0-10. Consider clarity, specificity, and usefulness."
    )
    print(f"   Original Score: {original_eval['loss']}")
    
    # Optimize
    print("\n🔄 Optimizing prompt...")
    result = optimizer.optimize(
        initial_text=initial_prompt,
        loss_fn=loss_fn,
        role_description="prompt for an AI writing assistant"
    )
    
    optimized_prompt = result['optimized_text']
    print(f"\n✨ Optimized Prompt:")
    print(f"   '{optimized_prompt}'")
    
    # Evaluate optimized
    optimized_eval = evaluator.evaluate_text(
        optimized_prompt,
        loss_fn="Rate this prompt quality from 0-10. Consider clarity, specificity, and usefulness."
    )
    print(f"   Optimized Score: {optimized_eval['loss']}")
    
    # Compare
    comparison = evaluator.compare_texts(
        initial_prompt,
        optimized_prompt,
        loss_fn="Rate this prompt quality from 0-10. Consider clarity, specificity, and usefulness."
    )
    
    print(f"\n📊 Comparison Results:")
    print(f"   Original Loss: {comparison['original_loss']}")
    print(f"   Optimized Loss: {comparison['optimized_loss']}")
    print(f"   Improved: {comparison['improvement']}")
    if comparison.get('improvement_ratio'):
        print(f"   Improvement Ratio: {comparison['improvement_ratio']:.2%}")
    
    # Test 2: Math problem solving prompt
    print("\n\n[Test 2] Math Problem Solving Prompt")
    print("-" * 70)
    
    initial_math_prompt = "Solve this math problem."
    math_loss_fn = tg.TextLoss(
        "Evaluate this prompt for solving math problems. A good prompt should: "
        "1) Specify the type of problem, 2) Ask for step-by-step solution, "
        "3) Request verification. Rate from 0-10."
    )
    
    print(f"\n📝 Original Prompt:")
    print(f"   '{initial_math_prompt}'")
    original_math_eval = evaluator.evaluate_text(
        initial_math_prompt,
        loss_fn="Rate this math problem solving prompt from 0-10."
    )
    print(f"   Original Score: {original_math_eval['loss']}")
    
    print("\n🔄 Optimizing prompt...")
    math_result = optimizer.optimize(
        initial_text=initial_math_prompt,
        loss_fn=math_loss_fn,
        role_description="prompt for solving math problems"
    )
    
    optimized_math_prompt = math_result['optimized_text']
    print(f"\n✨ Optimized Prompt:")
    print(f"   '{optimized_math_prompt}'")
    
    optimized_math_eval = evaluator.evaluate_text(
        optimized_math_prompt,
        loss_fn="Rate this math problem solving prompt from 0-10."
    )
    print(f"   Optimized Score: {optimized_math_eval['loss']}")
    
    math_comparison = evaluator.compare_texts(
        initial_math_prompt,
        optimized_math_prompt,
        loss_fn="Rate this math problem solving prompt from 0-10."
    )
    
    print(f"\n📊 Comparison Results:")
    print(f"   Original Loss: {math_comparison['original_loss']}")
    print(f"   Optimized Loss: {math_comparison['optimized_loss']}")
    print(f"   Improved: {math_comparison['improvement']}")
    if math_comparison.get('improvement_ratio'):
        print(f"   Improvement Ratio: {math_comparison['improvement_ratio']:.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Test 1 - Simple Prompt: {'✓ Improved' if comparison['improvement'] else '✗ Not Improved'}")
    print(f"Test 2 - Math Prompt: {'✓ Improved' if math_comparison['improvement'] else '✗ Not Improved'}")
    
    return comparison['improvement'] and math_comparison['improvement']

if __name__ == "__main__":
    try:
        success = test_prompt_optimization_effectiveness()
        if success:
            print("\n🎉 All optimization tests show improvement!")
        else:
            print("\n⚠️  Some optimizations did not show improvement.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
