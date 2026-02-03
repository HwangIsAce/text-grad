"""
Tests for TextGradOptimizer module.
"""

import pytest
from text_grad_module.config import TextGradConfig
from text_grad_module.optimizer import TextGradOptimizer


class TestTextGradConfig:
    """Tests for TextGradConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TextGradConfig()
        assert config.backward_engine == "gpt-4o"
        assert config.max_iterations == 1
        assert config.learning_rate == 1.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TextGradConfig(
            backward_engine="gpt-3.5-turbo",
            max_iterations=3,
            learning_rate=0.5
        )
        assert config.backward_engine == "gpt-3.5-turbo"
        assert config.max_iterations == 3
        assert config.learning_rate == 0.5
    
    def test_config_validation_max_iterations(self):
        """Test that max_iterations validation works."""
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            TextGradConfig(max_iterations=0)
    
    def test_config_validation_learning_rate(self):
        """Test that learning_rate validation works."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TextGradConfig(learning_rate=0)
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TextGradConfig(learning_rate=-1.0)
    
    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("TEXTGRAD_BACKWARD_ENGINE", "gpt-3.5-turbo")
        monkeypatch.setenv("TEXTGRAD_MAX_ITERATIONS", "5")
        monkeypatch.setenv("TEXTGRAD_LEARNING_RATE", "0.8")
        
        config = TextGradConfig.from_env()
        assert config.backward_engine == "gpt-3.5-turbo"
        assert config.max_iterations == 5
        assert config.learning_rate == 0.8


class TestTextGradOptimizer:
    """Tests for TextGradOptimizer."""
    
    def test_optimizer_initialization_default(self):
        """Test optimizer initialization with default config."""
        optimizer = TextGradOptimizer()
        assert optimizer.config is not None
        assert isinstance(optimizer.config, TextGradConfig)
    
    def test_optimizer_initialization_custom(self):
        """Test optimizer initialization with custom config."""
        config = TextGradConfig(backward_engine="gpt-3.5-turbo")
        optimizer = TextGradOptimizer(config)
        assert optimizer.config.backward_engine == "gpt-3.5-turbo"
    
    def test_create_variable(self):
        """Test variable creation."""
        optimizer = TextGradOptimizer()
        variable = optimizer.create_variable(
            "test text",
            requires_grad=True,
            role_description="test variable"
        )
        assert variable is not None
        assert variable.value == "test text"
        assert variable.requires_grad is True
        assert len(optimizer._variables) == 1
    
    def test_create_variable_without_grad(self):
        """Test variable creation without gradient."""
        optimizer = TextGradOptimizer()
        variable = optimizer.create_variable(
            "test text",
            requires_grad=False
        )
        assert variable.requires_grad is False
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        optimizer = TextGradOptimizer()
        variable = optimizer.create_variable("test", requires_grad=True)
        tg_optimizer = optimizer.create_optimizer([variable])
        assert tg_optimizer is not None
        assert optimizer._optimizer is not None
    
    def test_create_optimizer_auto_parameters(self):
        """Test optimizer creation with automatic parameter detection."""
        optimizer = TextGradOptimizer()
        var1 = optimizer.create_variable("test1", requires_grad=True)
        var2 = optimizer.create_variable("test2", requires_grad=False)
        var3 = optimizer.create_variable("test3", requires_grad=True)
        
        tg_optimizer = optimizer.create_optimizer()
        # Should only include variables with requires_grad=True
        assert len(tg_optimizer.parameters) == 2
    
    def test_reset(self):
        """Test optimizer reset."""
        optimizer = TextGradOptimizer()
        optimizer.create_variable("test1")
        optimizer.create_variable("test2")
        optimizer.create_optimizer()
        
        assert len(optimizer._variables) == 2
        assert optimizer._optimizer is not None
        
        optimizer.reset()
        
        assert len(optimizer._variables) == 0
        assert optimizer._optimizer is None


@pytest.mark.integration
class TestTextGradOptimizerIntegration:
    """Integration tests for TextGradOptimizer (requires actual TextGrad)."""
    
    @pytest.mark.skip(reason="Requires API key and may incur costs")
    def test_optimize_basic(self):
        """Test basic optimization (requires actual TextGrad API)."""
        import textgrad as tg
        
        config = TextGradConfig(max_iterations=1)
        optimizer = TextGradOptimizer(config)
        
        initial_text = "This is a test."
        loss_fn = tg.TextLoss("Evaluate if this text is clear and concise.")
        
        result = optimizer.optimize(
            initial_text=initial_text,
            loss_fn=loss_fn,
            role_description="test text"
        )
        
        assert "optimized_text" in result
        assert "iterations" in result
        assert "history" in result
        assert result["iterations"] == 1
        assert len(result["history"]) == 1
        assert result["history"][0]["iteration"] == 0
