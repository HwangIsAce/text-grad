"""
Tests for TextGradEvaluator module.
"""

import pytest
from text_grad_module.config import TextGradConfig
from text_grad_module.evaluator import TextGradEvaluator


class TestTextGradEvaluator:
    """Tests for TextGradEvaluator."""
    
    def test_evaluator_initialization_default(self):
        """Test evaluator initialization with default config."""
        evaluator = TextGradEvaluator()
        assert evaluator.config is not None
        assert isinstance(evaluator.config, TextGradConfig)
    
    def test_evaluator_initialization_custom(self):
        """Test evaluator initialization with custom config."""
        config = TextGradConfig(backward_engine="gpt-3.5-turbo")
        evaluator = TextGradEvaluator(config)
        assert evaluator.config.backward_engine == "gpt-3.5-turbo"
    
    def test_create_loss_fn(self):
        """Test loss function creation."""
        evaluator = TextGradEvaluator()
        loss_fn = evaluator.create_loss_fn("Evaluate the quality of the text.")
        assert loss_fn is not None
    
    def test_evaluate_text_with_string_prompt(self):
        """Test text evaluation with string prompt."""
        evaluator = TextGradEvaluator()
        # This will create a loss function internally
        result = evaluator.evaluate_text(
            text="This is a test.",
            loss_fn="Evaluate if this text is clear."
        )
        
        assert "loss" in result
        assert "text" in result
        assert result["text"] == "This is a test."
    
    def test_evaluate_text_with_ground_truth_match(self):
        """Test text evaluation with matching ground truth."""
        evaluator = TextGradEvaluator()
        result = evaluator.evaluate_text(
            text="Hello",
            loss_fn="Evaluate the text.",
            ground_truth="Hello"
        )
        
        assert "ground_truth" in result
        assert "matches" in result
        assert result["matches"] is True
    
    def test_evaluate_text_with_ground_truth_mismatch(self):
        """Test text evaluation with non-matching ground truth."""
        evaluator = TextGradEvaluator()
        result = evaluator.evaluate_text(
            text="Hello",
            loss_fn="Evaluate the text.",
            ground_truth="World"
        )
        
        assert "matches" in result
        assert result["matches"] is False
    
    def test_evaluate_text_with_ground_truth_case_insensitive(self):
        """Test that ground truth comparison is case-insensitive."""
        evaluator = TextGradEvaluator()
        result = evaluator.evaluate_text(
            text="Hello",
            loss_fn="Evaluate the text.",
            ground_truth="HELLO"
        )
        
        assert result["matches"] is True
    
    def test_compare_texts_basic(self):
        """Test basic text comparison."""
        evaluator = TextGradEvaluator()
        result = evaluator.compare_texts(
            original="Original text",
            optimized="Optimized text",
            loss_fn="Evaluate the text quality."
        )
        
        assert "original_loss" in result
        assert "optimized_loss" in result
        assert "improvement" in result
        assert "original_text" in result
        assert "optimized_text" in result
        assert result["original_text"] == "Original text"
        assert result["optimized_text"] == "Optimized text"
    
    def test_compare_texts_structure(self):
        """Test that compare_texts returns correct structure."""
        evaluator = TextGradEvaluator()
        result = evaluator.compare_texts(
            original="test1",
            optimized="test2",
            loss_fn="Evaluate."
        )
        
        required_keys = [
            "original_loss",
            "optimized_loss",
            "improvement",
            "original_text",
            "optimized_text"
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


@pytest.mark.integration
class TestTextGradEvaluatorIntegration:
    """Integration tests for TextGradEvaluator (requires actual TextGrad)."""
    
    @pytest.mark.skip(reason="Requires API key and may incur costs")
    def test_evaluate_text_integration(self):
        """Test text evaluation with actual TextGrad (requires API key)."""
        import textgrad as tg
        
        evaluator = TextGradEvaluator()
        loss_fn = tg.TextLoss("Rate the clarity of this text from 0 to 10.")
        
        result = evaluator.evaluate_text(
            text="This is a clear and concise sentence.",
            loss_fn=loss_fn
        )
        
        assert "loss" in result
        assert "text" in result
    
    @pytest.mark.skip(reason="Requires API key and may incur costs")
    def test_compare_texts_integration(self):
        """Test text comparison with actual TextGrad (requires API key)."""
        import textgrad as tg
        
        evaluator = TextGradEvaluator()
        loss_fn = tg.TextLoss("Rate the quality of this text from 0 to 10.")
        
        result = evaluator.compare_texts(
            original="Bad text.",
            optimized="This is a much better and more detailed text.",
            loss_fn=loss_fn
        )
        
        assert "original_loss" in result
        assert "optimized_loss" in result
        assert "improvement" in result
