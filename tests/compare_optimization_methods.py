"""
Compare GPT-4o direct optimization vs TextGrad optimization.
"""

import sys
from pathlib import Path
import time
from typing import Dict, List, Any

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from text_grad_module import TextGradConfig, TextGradOptimizer, TextGradEvaluator
import textgrad as tg
from openai import OpenAI

# Initialize OpenAI client for direct GPT-4o usage
client = OpenAI()


def optimize_with_gpt4o_direct(initial_prompt: str) -> Dict[str, Any]:
    """
    Optimize prompt using GPT-4o directly.
    
    Args:
        initial_prompt: Initial prompt to optimize
        
    Returns:
        Dictionary with optimization results
    """
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at improving prompts. Make them more specific, clear, and actionable."
                },
                {
                    "role": "user",
                    "content": f"Improve this prompt to make it more effective, specific, and actionable:\n\n{initial_prompt}\n\nProvide only the improved prompt, no explanation."
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        optimized_prompt = response.choices[0].message.content.strip()
        elapsed_time = time.time() - start_time
        
        return {
            "optimized_text": optimized_prompt,
            "method": "GPT-4o Direct",
            "time": elapsed_time,
            "api_calls": 1,
            "success": True
        }
    except Exception as e:
        return {
            "optimized_text": initial_prompt,
            "method": "GPT-4o Direct",
            "time": time.time() - start_time,
            "api_calls": 1,
            "success": False,
            "error": str(e)
        }


def optimize_with_textgrad(initial_prompt: str, role_description: str = "prompt") -> Dict[str, Any]:
    """
    Optimize prompt using TextGrad.
    
    Args:
        initial_prompt: Initial prompt to optimize
        role_description: Description of the prompt's role
        
    Returns:
        Dictionary with optimization results
    """
    start_time = time.time()
    
    try:
        config = TextGradConfig(backward_engine="gpt-4o", max_iterations=1)
        optimizer = TextGradOptimizer(config)
        
        loss_fn = tg.TextLoss(
            "Evaluate this prompt. A good prompt should be: "
            "1) Clear and specific, 2) Provide context, 3) Include examples if needed, "
            "4) Be actionable. Improve the prompt to make it more effective."
        )
        
        result = optimizer.optimize(
            initial_text=initial_prompt,
            loss_fn=loss_fn,
            role_description=role_description
        )
        
        elapsed_time = time.time() - start_time
        
        # Count API calls: 1 forward (loss) + 1 backward (gradient) + 1 step (optimization)
        # Actually, TextGrad makes multiple calls internally
        api_calls = result.get("iterations", 1) * 3  # Rough estimate
        
        return {
            "optimized_text": result["optimized_text"],
            "method": "TextGrad",
            "time": elapsed_time,
            "api_calls": api_calls,
            "iterations": result.get("iterations", 1),
            "success": True
        }
    except Exception as e:
        return {
            "optimized_text": initial_prompt,
            "method": "TextGrad",
            "time": time.time() - start_time,
            "api_calls": 0,
            "success": False,
            "error": str(e)
        }


def evaluate_prompt_quality(prompt: str, evaluator: TextGradEvaluator) -> Dict[str, Any]:
    """
    Evaluate prompt quality using multiple criteria.
    
    Args:
        prompt: Prompt to evaluate
        evaluator: TextGradEvaluator instance
        
    Returns:
        Dictionary with evaluation scores
    """
    evaluation_criteria = [
        ("specificity", "Rate how specific and detailed this prompt is. Respond with only a number from 0 to 10."),
        ("actionability", "Rate how actionable and clear the instructions are. Respond with only a number from 0 to 10."),
        ("completeness", "Rate how complete and comprehensive this prompt is. Respond with only a number from 0 to 10."),
        ("clarity", "Rate how clear and easy to understand this prompt is. Respond with only a number from 0 to 10."),
    ]
    
    scores = {}
    for criterion, prompt_text in evaluation_criteria:
        try:
            result = evaluator.evaluate_text(
                text=prompt,
                loss_fn=prompt_text,
                role_description="prompt to evaluate"
            )
            # Try to extract numeric score
            loss_value = result.get("loss", "0")
            
            # Try to extract number from response
            import re
            # Look for numbers in the response
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', str(loss_value))
            if numbers:
                # Take the first number found, but ensure it's between 0-10
                score = float(numbers[0])
                if score > 10:
                    score = 10
                scores[criterion] = score
            else:
                # Fallback: use length and content as proxy
                # More specific prompts tend to be longer and have more details
                if criterion == "specificity":
                    scores[criterion] = min(10, len(prompt.split()) / 5)
                elif criterion == "actionability":
                    action_words = ["create", "write", "solve", "explain", "analyze", "design", "build"]
                    scores[criterion] = min(10, sum(1 for word in action_words if word in prompt.lower()) * 2)
                elif criterion == "completeness":
                    scores[criterion] = min(10, len(prompt.split()) / 3)
                else:  # clarity
                    scores[criterion] = min(10, len(prompt.split()) / 4)
        except Exception as e:
            # Fallback scoring
            if criterion == "specificity":
                scores[criterion] = min(10, len(prompt.split()) / 5)
            elif criterion == "actionability":
                action_words = ["create", "write", "solve", "explain", "analyze", "design", "build"]
                scores[criterion] = min(10, sum(1 for word in action_words if word in prompt.lower()) * 2)
            elif criterion == "completeness":
                scores[criterion] = min(10, len(prompt.split()) / 3)
            else:  # clarity
                scores[criterion] = min(10, len(prompt.split()) / 4)
    
    # Calculate overall score
    if scores:
        scores["overall"] = sum(scores.values()) / len(scores)
    else:
        scores["overall"] = 0
    
    return scores


def compare_methods(initial_prompts: List[Dict[str, str]]) -> None:
    """
    Compare GPT-4o direct vs TextGrad optimization methods.
    
    Args:
        initial_prompts: List of test prompts with descriptions
    """
    print("=" * 80)
    print("GPT-4o Direct vs TextGrad Optimization Comparison")
    print("=" * 80)
    
    evaluator = TextGradEvaluator()
    results = []
    
    for i, test_case in enumerate(initial_prompts, 1):
        initial_prompt = test_case["prompt"]
        description = test_case.get("description", f"Test {i}")
        
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {description}")
        print(f"{'=' * 80}")
        print(f"\n📝 Initial Prompt:")
        print(f"   '{initial_prompt}'")
        
        # Method 1: GPT-4o Direct
        print(f"\n🔄 Method 1: GPT-4o Direct Optimization...")
        gpt4o_result = optimize_with_gpt4o_direct(initial_prompt)
        
        if gpt4o_result["success"]:
            print(f"   ✓ Optimized in {gpt4o_result['time']:.2f}s")
            print(f"   API Calls: {gpt4o_result['api_calls']}")
            print(f"\n   Result:")
            print(f"   '{gpt4o_result['optimized_text']}'")
            
            # Evaluate
            print(f"\n   📊 Evaluating quality...")
            gpt4o_scores = evaluate_prompt_quality(gpt4o_result['optimized_text'], evaluator)
            print(f"   Specificity: {gpt4o_scores.get('specificity', 'N/A')}")
            print(f"   Actionability: {gpt4o_scores.get('actionability', 'N/A')}")
            print(f"   Completeness: {gpt4o_scores.get('completeness', 'N/A')}")
            print(f"   Clarity: {gpt4o_scores.get('clarity', 'N/A')}")
            print(f"   Overall: {gpt4o_scores.get('overall', 'N/A'):.2f}")
        else:
            print(f"   ✗ Failed: {gpt4o_result.get('error', 'Unknown error')}")
            gpt4o_scores = {}
        
        # Method 2: TextGrad
        print(f"\n🔄 Method 2: TextGrad Optimization...")
        textgrad_result = optimize_with_textgrad(initial_prompt, test_case.get("role", "prompt"))
        
        if textgrad_result["success"]:
            print(f"   ✓ Optimized in {textgrad_result['time']:.2f}s")
            print(f"   API Calls: {textgrad_result['api_calls']} (estimated)")
            print(f"   Iterations: {textgrad_result.get('iterations', 1)}")
            print(f"\n   Result:")
            print(f"   '{textgrad_result['optimized_text']}'")
            
            # Evaluate
            print(f"\n   📊 Evaluating quality...")
            textgrad_scores = evaluate_prompt_quality(textgrad_result['optimized_text'], evaluator)
            print(f"   Specificity: {textgrad_scores.get('specificity', 'N/A')}")
            print(f"   Actionability: {textgrad_scores.get('actionability', 'N/A')}")
            print(f"   Completeness: {textgrad_scores.get('completeness', 'N/A')}")
            print(f"   Clarity: {textgrad_scores.get('clarity', 'N/A')}")
            print(f"   Overall: {textgrad_scores.get('overall', 'N/A'):.2f}")
        else:
            print(f"   ✗ Failed: {textgrad_result.get('error', 'Unknown error')}")
            textgrad_scores = {}
        
        # Comparison
        print(f"\n📈 Comparison Summary:")
        if gpt4o_result["success"] and textgrad_result["success"]:
            print(f"   Time: GPT-4o ({gpt4o_result['time']:.2f}s) vs TextGrad ({textgrad_result['time']:.2f}s)")
            print(f"   API Calls: GPT-4o ({gpt4o_result['api_calls']}) vs TextGrad ({textgrad_result['api_calls']})")
            
            gpt4o_overall = gpt4o_scores.get('overall', 0)
            textgrad_overall = textgrad_scores.get('overall', 0)
            
            if gpt4o_overall > textgrad_overall:
                winner = "GPT-4o Direct"
                diff = gpt4o_overall - textgrad_overall
            elif textgrad_overall > gpt4o_overall:
                winner = "TextGrad"
                diff = textgrad_overall - gpt4o_overall
            else:
                winner = "Tie"
                diff = 0
            
            print(f"   Quality Score: GPT-4o ({gpt4o_overall:.2f}) vs TextGrad ({textgrad_overall:.2f})")
            print(f"   Winner: {winner} (+{diff:.2f})")
        
        results.append({
            "test_case": description,
            "initial_prompt": initial_prompt,
            "gpt4o": gpt4o_result,
            "textgrad": textgrad_result,
            "gpt4o_scores": gpt4o_scores,
            "textgrad_scores": textgrad_scores
        })
    
    # Final Summary
    print(f"\n\n{'=' * 80}")
    print("Final Summary")
    print(f"{'=' * 80}")
    
    successful_tests = [r for r in results if r["gpt4o"]["success"] and r["textgrad"]["success"]]
    
    if successful_tests:
        avg_gpt4o_time = sum(r["gpt4o"]["time"] for r in successful_tests) / len(successful_tests)
        avg_textgrad_time = sum(r["textgrad"]["time"] for r in successful_tests) / len(successful_tests)
        avg_gpt4o_calls = sum(r["gpt4o"]["api_calls"] for r in successful_tests) / len(successful_tests)
        avg_textgrad_calls = sum(r["textgrad"]["api_calls"] for r in successful_tests) / len(successful_tests)
        avg_gpt4o_score = sum(r["gpt4o_scores"].get("overall", 0) for r in successful_tests) / len(successful_tests)
        avg_textgrad_score = sum(r["textgrad_scores"].get("overall", 0) for r in successful_tests) / len(successful_tests)
        
        print(f"\nAverage Metrics ({len(successful_tests)} successful tests):")
        print(f"  Time: GPT-4o ({avg_gpt4o_time:.2f}s) vs TextGrad ({avg_textgrad_time:.2f}s)")
        print(f"  API Calls: GPT-4o ({avg_gpt4o_calls:.1f}) vs TextGrad ({avg_textgrad_calls:.1f})")
        print(f"  Quality Score: GPT-4o ({avg_gpt4o_score:.2f}) vs TextGrad ({avg_textgrad_score:.2f})")
        
        print(f"\nRecommendations:")
        if avg_gpt4o_time < avg_textgrad_time:
            print(f"  ⚡ Speed: GPT-4o Direct is faster")
        else:
            print(f"  ⚡ Speed: TextGrad is faster")
        
        if avg_gpt4o_calls < avg_textgrad_calls:
            print(f"  💰 Cost: GPT-4o Direct uses fewer API calls")
        else:
            print(f"  💰 Cost: TextGrad uses fewer API calls")
        
        if avg_gpt4o_score > avg_textgrad_score:
            print(f"  ✨ Quality: GPT-4o Direct produces better results")
        elif avg_textgrad_score > avg_gpt4o_score:
            print(f"  ✨ Quality: TextGrad produces better results")
        else:
            print(f"  ✨ Quality: Both methods produce similar results")


if __name__ == "__main__":
    # Test cases
    test_prompts = [
        {
            "prompt": "Write a story.",
            "description": "Simple Story Writing",
            "role": "prompt for story writing"
        },
        {
            "prompt": "Solve this math problem.",
            "description": "Math Problem Solving",
            "role": "prompt for solving math problems"
        },
        {
            "prompt": "Explain quantum physics.",
            "description": "Educational Explanation",
            "role": "prompt for educational content"
        }
    ]
    
    try:
        compare_methods(test_prompts)
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
