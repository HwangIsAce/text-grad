"""
Quick test script to verify TextGrad module functionality.
"""

import sys
sys.path.insert(0, 'src')

from text_grad_module import TextGradConfig, TextGradOptimizer, TextGradEvaluator
import textgrad as tg

def test_config():
    """Test configuration."""
    print("=== Testing Config ===")
    config = TextGradConfig(backward_engine="gpt-4o", max_iterations=1)
    print(f"✓ Config created: backward_engine={config.backward_engine}, max_iterations={config.max_iterations}")
    return config

def test_optimizer_initialization():
    """Test optimizer initialization."""
    print("\n=== Testing Optimizer Initialization ===")
    config = TextGradConfig(backward_engine="gpt-4o", max_iterations=1)
    optimizer = TextGradOptimizer(config)
    print("✓ Optimizer initialized successfully")
    return optimizer

def test_evaluator():
    """Test evaluator."""
    print("\n=== Testing Evaluator ===")
    evaluator = TextGradEvaluator()
    print("✓ Evaluator initialized successfully")
    return evaluator

def test_simple_optimization():
    """Test a simple text optimization."""
    print("\n=== Testing Simple Text Optimization ===")
    try:
        config = TextGradConfig(backward_engine="gpt-4o", max_iterations=1)
        optimizer = TextGradOptimizer(config)
        
        initial_text = "This is a test sentence."
        loss_fn = tg.TextLoss("Evaluate if this text is clear and concise.")
        
        print(f"Original text: {initial_text}")
        print("Running optimization...")
        
        result = optimizer.optimize(
            initial_text=initial_text,
            loss_fn=loss_fn,
            role_description="test text"
        )
        
        print(f"✓ Optimization completed!")
        print(f"Optimized text: {result['optimized_text']}")
        print(f"Iterations: {result['iterations']}")
        
        return True
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return False

def test_text_evaluation():
    """Test text evaluation."""
    print("\n=== Testing Text Evaluation ===")
    try:
        evaluator = TextGradEvaluator()
        
        text = "This is a well-written sentence."
        result = evaluator.evaluate_text(
            text=text,
            loss_fn="Rate the quality of this text from 0 to 10.",
        )
        
        print(f"✓ Evaluation completed!")
        print(f"Text: {text}")
        print(f"Loss: {result['loss']}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("TextGrad Module Test\n")
    print("=" * 50)
    
    # Test 1: Config
    try:
        test_config()
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        sys.exit(1)
    
    # Test 2: Optimizer initialization
    try:
        test_optimizer_initialization()
    except Exception as e:
        print(f"✗ Optimizer initialization failed: {e}")
        print("This might be due to missing OPENAI_API_KEY")
        sys.exit(1)
    
    # Test 3: Evaluator
    try:
        test_evaluator()
    except Exception as e:
        print(f"✗ Evaluator test failed: {e}")
        sys.exit(1)
    
    # Test 4: Simple optimization (requires API key)
    print("\n" + "=" * 50)
    print("Running optimization test (this will make API calls)...")
    opt_success = test_simple_optimization()
    
    # Test 5: Text evaluation (requires API key)
    print("\n" + "=" * 50)
    print("Running evaluation test (this will make API calls)...")
    eval_success = test_text_evaluation()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Config: ✓")
    print(f"  Optimizer Init: ✓")
    print(f"  Evaluator Init: ✓")
    print(f"  Optimization: {'✓' if opt_success else '✗'}")
    print(f"  Evaluation: {'✓' if eval_success else '✗'}")
    
    if opt_success and eval_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check API key and network connection.")
