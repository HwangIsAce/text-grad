"""
Objective comparison between TextGrad and GPT-4o direct optimization.
Uses LLM-based evaluation and statistical analysis.
"""

import sys
from pathlib import Path
import time
from typing import Dict, List, Any
import statistics

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from text_grad_module import TextGradConfig, TextGradOptimizer, TextGradEvaluator
import textgrad as tg
from openai import OpenAI

client = OpenAI()


def evaluate_with_llm(prompt: str, evaluation_criteria: str) -> float:
    """
    Use LLM to objectively evaluate prompt quality.
    
    Args:
        prompt: Prompt to evaluate
        evaluation_criteria: What to evaluate
        
    Returns:
        Score from 0-10
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at evaluating prompts. Rate prompts objectively from 0 to 10."
                },
                {
                    "role": "user",
                    "content": f"""Evaluate this prompt based on: {evaluation_criteria}

Prompt: "{prompt}"

Rate the prompt from 0 to 10. Respond with ONLY a number between 0 and 10, no explanation."""
                }
            ],
            temperature=0.0,  # Deterministic
            max_tokens=10
        )
        
        score_text = response.choices[0].message.content.strip()
        # Extract number
        import re
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', score_text)
        if numbers:
            score = float(numbers[0])
            return min(10, max(0, score))
        return 0.0
    except Exception as e:
        print(f"   Evaluation error: {e}")
        return 0.0


def evaluate_multiple_criteria(prompt: str) -> Dict[str, float]:
    """
    Evaluate prompt on multiple objective criteria.
    
    Args:
        prompt: Prompt to evaluate
        
    Returns:
        Dictionary with scores for each criterion
    """
    criteria = {
        "specificity": "How specific and detailed is the prompt? (0-10)",
        "clarity": "How clear and easy to understand is the prompt? (0-10)",
        "actionability": "How actionable and executable is the prompt? (0-10)",
        "completeness": "How complete and comprehensive is the prompt? (0-10)"
    }
    
    scores = {}
    for criterion, description in criteria.items():
        scores[criterion] = evaluate_with_llm(prompt, description)
        time.sleep(0.2)  # Rate limiting
    
    scores["overall"] = statistics.mean(list(scores.values()))
    return scores


def test_textgrad_gradual_improvement_objective():
    """Test TextGrad with objective LLM-based evaluation."""
    print("=" * 80)
    print("TextGrad - Objective Evaluation (LLM-based)")
    print("=" * 80)
    
    initial_prompt = "Write a story."
    config = TextGradConfig(backward_engine="gpt-4o", max_iterations=3)
    optimizer = TextGradOptimizer(config)
    
    loss_fn = tg.TextLoss(
        "Evaluate this prompt. A good prompt should be: "
        "1) Clear and specific, 2) Provide context, 3) Include examples if needed, "
        "4) Be actionable. Identify what needs improvement."
    )
    
    print(f"\n📝 Initial Prompt: '{initial_prompt}'")
    
    # Evaluate initial prompt objectively
    print("\n📊 Evaluating initial prompt...")
    initial_scores = evaluate_multiple_criteria(initial_prompt)
    print(f"   Specificity: {initial_scores['specificity']:.2f}")
    print(f"   Clarity: {initial_scores['clarity']:.2f}")
    print(f"   Actionability: {initial_scores['actionability']:.2f}")
    print(f"   Completeness: {initial_scores['completeness']:.2f}")
    print(f"   Overall: {initial_scores['overall']:.2f}/10")
    
    # Track iterations
    variable = optimizer.create_variable(
        initial_prompt,
        requires_grad=True,
        role_description="prompt for story writing"
    )
    
    optimizer_instance = optimizer.create_optimizer([variable])
    
    iteration_scores = [initial_scores]
    
    for iteration in range(1, config.max_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}")
        print(f"{'=' * 60}")
        
        # Forward pass
        loss = loss_fn(variable)
        
        # Backward pass
        loss.backward()
        
        # Step
        optimizer_instance.step()
        
        # Evaluate after step
        print(f"   Current prompt: '{variable.value[:100]}...'")
        print(f"   Evaluating...")
        current_scores = evaluate_multiple_criteria(variable.value)
        print(f"   Specificity: {current_scores['specificity']:.2f} (Δ{current_scores['specificity'] - iteration_scores[-1]['specificity']:+.2f})")
        print(f"   Clarity: {current_scores['clarity']:.2f} (Δ{current_scores['clarity'] - iteration_scores[-1]['clarity']:+.2f})")
        print(f"   Actionability: {current_scores['actionability']:.2f} (Δ{current_scores['actionability'] - iteration_scores[-1]['actionability']:+.2f})")
        print(f"   Completeness: {current_scores['completeness']:.2f} (Δ{current_scores['completeness'] - iteration_scores[-1]['completeness']:+.2f})")
        print(f"   Overall: {current_scores['overall']:.2f}/10 (Δ{current_scores['overall'] - iteration_scores[-1]['overall']:+.2f})")
        
        iteration_scores.append(current_scores)
    
    # Analyze progression
    print(f"\n{'=' * 80}")
    print("Progression Analysis")
    print(f"{'=' * 80}")
    
    overall_scores = [s['overall'] for s in iteration_scores]
    print(f"\nOverall Score Progression:")
    for i, score in enumerate(overall_scores):
        if i == 0:
            print(f"   Initial: {score:.2f}")
        else:
            improvement = score - overall_scores[i-1]
            print(f"   Iteration {i}: {score:.2f} (Δ{improvement:+.2f})")
    
    # Check if gradual improvement
    is_gradual = all(overall_scores[i] <= overall_scores[i+1] for i in range(len(overall_scores)-1))
    total_improvement = overall_scores[-1] - overall_scores[0]
    
    print(f"\n✅ Gradual Improvement: {'Yes' if is_gradual else 'No'}")
    print(f"   Total Improvement: {total_improvement:+.2f} points")
    print(f"   Improvement Rate: {(total_improvement / overall_scores[0] * 100) if overall_scores[0] > 0 else 0:.1f}%")
    
    return {
        "initial": initial_scores,
        "final": iteration_scores[-1],
        "progression": iteration_scores,
        "is_gradual": is_gradual,
        "total_improvement": total_improvement
    }


def test_gpt4o_consistency_objective():
    """Test GPT-4o direct optimization with objective evaluation."""
    print(f"\n\n{'=' * 80}")
    print("GPT-4o Direct - Objective Evaluation (Multiple Runs)")
    print(f"{'=' * 80}")
    
    initial_prompt = "Write a story."
    
    print(f"\n📝 Initial Prompt: '{initial_prompt}'")
    
    # Evaluate initial
    print("\n📊 Evaluating initial prompt...")
    initial_scores = evaluate_multiple_criteria(initial_prompt)
    print(f"   Overall: {initial_scores['overall']:.2f}/10")
    
    # Run optimization multiple times
    print(f"\n🔄 Running GPT-4o optimization 5 times...")
    results = []
    
    for run in range(5):
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
                        "content": f"Improve this prompt to make it more effective, specific, and actionable:\n\n{initial_prompt}\n\nProvide only the improved prompt, no explanation."
                    }
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            optimized = response.choices[0].message.content.strip()
            print(f"\n   Run {run + 1}: '{optimized[:80]}...'")
            
            # Evaluate objectively
            scores = evaluate_multiple_criteria(optimized)
            print(f"      Overall: {scores['overall']:.2f}/10")
            
            results.append({
                "run": run + 1,
                "prompt": optimized,
                "scores": scores
            })
            
            time.sleep(0.5)
        except Exception as e:
            print(f"   Run {run + 1} failed: {e}")
    
    # Statistical analysis
    if len(results) > 1:
        overall_scores = [r['scores']['overall'] for r in results]
        
        print(f"\n{'=' * 80}")
        print("Statistical Analysis")
        print(f"{'=' * 80}")
        print(f"\nOverall Scores: {[f'{s:.2f}' for s in overall_scores]}")
        print(f"   Mean: {statistics.mean(overall_scores):.2f}")
        print(f"   Median: {statistics.median(overall_scores):.2f}")
        print(f"   Std Dev: {statistics.stdev(overall_scores):.2f}")
        print(f"   Min: {min(overall_scores):.2f}")
        print(f"   Max: {max(overall_scores):.2f}")
        print(f"   Range: {max(overall_scores) - min(overall_scores):.2f}")
        
        # Coefficient of variation (CV) - measure of consistency
        cv = (statistics.stdev(overall_scores) / statistics.mean(overall_scores)) * 100 if statistics.mean(overall_scores) > 0 else 0
        print(f"   Coefficient of Variation: {cv:.1f}%")
        
        if cv < 10:
            consistency = "High"
        elif cv < 20:
            consistency = "Medium"
        else:
            consistency = "Low"
        
        print(f"\n   Consistency: {consistency}")
        print(f"   Average Improvement: {statistics.mean(overall_scores) - initial_scores['overall']:+.2f} points")
    
    return {
        "initial": initial_scores,
        "results": results,
        "statistics": {
            "mean": statistics.mean(overall_scores) if len(results) > 1 else 0,
            "std": statistics.stdev(overall_scores) if len(results) > 1 else 0,
            "cv": cv if len(results) > 1 else 0
        } if len(results) > 1 else {}
    }


def compare_objectively():
    """Compare both methods objectively."""
    print("=" * 80)
    print("OBJECTIVE COMPARISON: TextGrad vs GPT-4o Direct")
    print("=" * 80)
    
    # Test TextGrad
    textgrad_result = test_textgrad_gradual_improvement_objective()
    
    # Test GPT-4o
    gpt4o_result = test_gpt4o_consistency_objective()
    
    # Final comparison
    print(f"\n\n{'=' * 80}")
    print("FINAL OBJECTIVE COMPARISON")
    print(f"{'=' * 80}")
    
    print(f"\n📊 TextGrad Results:")
    print(f"   Initial Score: {textgrad_result['initial']['overall']:.2f}/10")
    print(f"   Final Score: {textgrad_result['final']['overall']:.2f}/10")
    print(f"   Total Improvement: {textgrad_result['total_improvement']:+.2f} points")
    print(f"   Gradual Improvement: {'Yes' if textgrad_result['is_gradual'] else 'No'}")
    
    if gpt4o_result.get('statistics'):
        print(f"\n📊 GPT-4o Direct Results:")
        print(f"   Initial Score: {gpt4o_result['initial']['overall']:.2f}/10")
        print(f"   Average Final Score: {gpt4o_result['statistics']['mean']:.2f}/10")
        print(f"   Std Deviation: {gpt4o_result['statistics']['std']:.2f}")
        print(f"   Consistency (CV): {gpt4o_result['statistics']['cv']:.1f}%")
        print(f"   Average Improvement: {gpt4o_result['statistics']['mean'] - gpt4o_result['initial']['overall']:+.2f} points")
        
        print(f"\n🎯 Key Findings:")
        print(f"   1. TextGrad shows {'gradual' if textgrad_result['is_gradual'] else 'non-gradual'} improvement")
        print(f"   2. GPT-4o Direct shows {'high' if gpt4o_result['statistics']['cv'] < 10 else 'medium' if gpt4o_result['statistics']['cv'] < 20 else 'low'} consistency")
        print(f"   3. TextGrad improvement: {textgrad_result['total_improvement']:.2f} points")
        print(f"   4. GPT-4o average improvement: {gpt4o_result['statistics']['mean'] - gpt4o_result['initial']['overall']:.2f} points")
        
        if textgrad_result['final']['overall'] > gpt4o_result['statistics']['mean']:
            print(f"\n   ✅ TextGrad produces higher quality prompts")
        elif gpt4o_result['statistics']['mean'] > textgrad_result['final']['overall']:
            print(f"\n   ✅ GPT-4o Direct produces higher quality prompts")
        else:
            print(f"\n   ⚖️  Both methods produce similar quality")


if __name__ == "__main__":
    try:
        compare_objectively()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
