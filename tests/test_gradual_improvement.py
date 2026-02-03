"""
Test to verify TextGrad improves prompts gradually vs GPT-4o direct which is random.
Compare consistency and gradual improvement patterns.
"""

import sys
from pathlib import Path
import time
from typing import Dict

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from text_grad_module import TextGradConfig, TextGradOptimizer, TextGradEvaluator
import textgrad as tg
from openai import OpenAI

client = OpenAI()


def evaluate_prompt_length_and_detail(prompt: str) -> Dict[str, float]:
    """
    Simple evaluation based on prompt characteristics.
    More detailed prompts should be longer and have more structure.
    """
    word_count = len(prompt.split())
    has_structure = sum(1 for marker in [":", "-", "1.", "2.", "3.", "*"] if marker in prompt)
    has_specifics = sum(1 for word in ["specific", "detailed", "include", "describe", "explain"] if word.lower() in prompt.lower())
    
    return {
        "word_count": word_count,
        "structure_score": min(10, has_structure * 2),
        "specificity_score": min(10, has_specifics * 2),
        "overall": min(10, (word_count / 10) + (has_structure * 1.5) + (has_specifics * 1.5))
    }


def test_textgrad_gradual_improvement():
    """Test if TextGrad improves gradually."""
    print("=" * 80)
    print("TextGrad Gradual Improvement Test")
    print("=" * 80)
    
    initial_prompt = "Write a story."
    config = TextGradConfig(backward_engine="gpt-4o", max_iterations=5)
    optimizer = TextGradOptimizer(config)
    
    loss_fn = tg.TextLoss(
        "Evaluate this prompt. A good prompt should be: "
        "1) Clear and specific, 2) Provide context, 3) Include examples if needed, "
        "4) Be actionable. Identify what needs improvement."
    )
    
    print(f"\n📝 Initial Prompt: '{initial_prompt}'")
    
    # Track each iteration
    variable = optimizer.create_variable(
        initial_prompt,
        requires_grad=True,
        role_description="prompt for story writing"
    )
    
    optimizer_instance = optimizer.create_optimizer([variable])
    
    iterations_data = []
    
    for iteration in range(config.max_iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}")
        print(f"{'=' * 60}")
        
        # Forward pass - evaluate current prompt
        loss = loss_fn(variable)
        
        # Evaluate current state
        current_eval = evaluate_prompt_length_and_detail(variable.value)
        
        print(f"Current Prompt: '{variable.value[:150]}...' (if longer)")
        print(f"Word Count: {current_eval['word_count']}")
        print(f"Structure Score: {current_eval['structure_score']:.2f}")
        print(f"Overall Score: {current_eval['overall']:.2f}")
        print(f"Loss: {str(loss.value)[:100] if hasattr(loss, 'value') else 'N/A'}...")
        
        iterations_data.append({
            "iteration": iteration + 1,
            "prompt": variable.value,
            "word_count": current_eval['word_count'],
            "score": current_eval['overall']
        })
        
        # Backward pass - get gradient (improvement direction)
        loss.backward()
        
        # Check if there are gradients
        if hasattr(variable, 'grads') and variable.grads:
            print(f"Gradient: {str(variable.grads)[:200]}...")
        
        # Step - apply improvement
        optimizer_instance.step()
        
        time.sleep(0.3)
    
    # Analyze progression
    print(f"\n{'=' * 80}")
    print("Progression Analysis")
    print(f"{'=' * 80}")
    
    print(f"\n📈 Word Count Progression:")
    for data in iterations_data:
        print(f"   Iteration {data['iteration']}: {data['word_count']} words (Score: {data['score']:.2f})")
    
    # Check if it's gradually improving
    word_counts = [d['word_count'] for d in iterations_data]
    scores = [d['score'] for d in iterations_data]
    
    is_gradual = all(word_counts[i] <= word_counts[i+1] for i in range(len(word_counts)-1))
    is_improving = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
    
    print(f"\n✅ Gradual Improvement: {'Yes' if is_gradual else 'No'}")
    print(f"   Word count increases: {word_counts}")
    print(f"   Score progression: {[f'{s:.2f}' for s in scores]}")
    
    if is_gradual and is_improving:
        print(f"\n✨ TextGrad shows GRADUAL improvement across iterations!")
    else:
        print(f"\n⚠️  Improvement pattern is not strictly gradual (may have plateaus)")


def test_gpt4o_randomness():
    """Test if GPT-4o direct optimization produces random/varying results."""
    print(f"\n\n{'=' * 80}")
    print("GPT-4o Direct Optimization - Consistency Test")
    print(f"{'=' * 80}")
    
    initial_prompt = "Write a story."
    
    print(f"\n📝 Initial Prompt: '{initial_prompt}'")
    print(f"\n🔄 Running GPT-4o optimization 3 times to check consistency...")
    
    results = []
    for run in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at improving prompts."
                    },
                    {
                        "role": "user",
                        "content": f"Improve this prompt: {initial_prompt}\n\nProvide only the improved prompt, no explanation."
                    }
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            optimized = response.choices[0].message.content.strip()
            eval_result = evaluate_prompt_length_and_detail(optimized)
            
            print(f"\nRun {run + 1}:")
            print(f"   '{optimized[:150]}...'")
            print(f"   Word Count: {eval_result['word_count']}, Score: {eval_result['overall']:.2f}")
            
            results.append({
                "run": run + 1,
                "prompt": optimized,
                "word_count": eval_result['word_count'],
                "score": eval_result['overall']
            })
        except Exception as e:
            print(f"   Error: {e}")
    
    # Check variance
    if len(results) > 1:
        word_counts = [r['word_count'] for r in results]
        scores = [r['score'] for r in results]
        
        word_variance = max(word_counts) - min(word_counts)
        score_variance = max(scores) - min(scores)
        
        print(f"\n📊 Consistency Analysis:")
        print(f"   Word Count Range: {min(word_counts)} - {max(word_counts)} (Variance: {word_variance})")
        print(f"   Score Range: {min(scores):.2f} - {max(scores):.2f} (Variance: {score_variance:.2f})")
        
        if word_variance > 20 or score_variance > 2:
            print(f"\n⚠️  GPT-4o shows HIGH VARIANCE - results are inconsistent/random")
        else:
            print(f"\n✅ GPT-4o shows LOW VARIANCE - results are relatively consistent")


if __name__ == "__main__":
    try:
        test_textgrad_gradual_improvement()
        test_gpt4o_randomness()
        
        print(f"\n\n{'=' * 80}")
        print("Conclusion")
        print(f"{'=' * 80}")
        print("""
TextGrad should show:
- ✅ Gradual improvement across iterations
- ✅ Consistent improvement direction
- ✅ Each iteration builds on the previous

GPT-4o Direct should show:
- ⚠️  Random/varying results each run
- ⚠️  No gradual progression (one-shot)
- ⚠️  Inconsistent quality
        """)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
