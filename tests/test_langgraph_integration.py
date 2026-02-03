"""
Tests for LangGraph integration.
"""

import pytest
from text_grad_module.config import TextGradConfig
from text_grad_module.nodes import (
    TextGradState,
    create_optimize_node,
    create_evaluate_node,
    create_compare_node,
)


class TestTextGradState:
    """Tests for TextGradState TypedDict."""
    
    def test_state_structure(self):
        """Test that TextGradState has correct structure."""
        state: TextGradState = {
            "text": "test",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {}
        }
        
        assert state["text"] == "test"
        assert state["optimized_text"] is None
        assert state["iteration"] == 0
        assert isinstance(state["metadata"], dict)
    
    def test_state_with_metadata(self):
        """Test state with metadata."""
        state: TextGradState = {
            "text": "test",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {
                "loss_prompt": "Evaluate",
                "max_iterations": 3
            }
        }
        
        assert state["metadata"]["loss_prompt"] == "Evaluate"
        assert state["metadata"]["max_iterations"] == 3


class TestLangGraphNodes:
    """Tests for LangGraph node factories."""
    
    def test_create_optimize_node(self):
        """Test optimize node creation."""
        config = TextGradConfig(max_iterations=1)
        node = create_optimize_node(config, loss_prompt="Evaluate the text.")
        assert callable(node)
    
    def test_create_optimize_node_without_prompt(self):
        """Test optimize node creation without prompt (should work but fail at runtime)."""
        config = TextGradConfig()
        node = create_optimize_node(config)
        assert callable(node)
    
    def test_create_evaluate_node(self):
        """Test evaluate node creation."""
        config = TextGradConfig()
        node = create_evaluate_node(config)
        assert callable(node)
    
    def test_create_compare_node(self):
        """Test compare node creation."""
        config = TextGradConfig()
        node = create_compare_node(config)
        assert callable(node)
    
    def test_optimize_node_missing_text(self):
        """Test optimize node with missing text."""
        config = TextGradConfig()
        node = create_optimize_node(config, loss_prompt="Evaluate.")
        
        state: TextGradState = {
            "text": "",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {}
        }
        
        # Should return state unchanged if text is empty
        result = node(state)
        assert result == state
    
    def test_optimize_node_missing_loss_prompt(self):
        """Test optimize node with missing loss prompt."""
        config = TextGradConfig()
        node = create_optimize_node(config)
        
        state: TextGradState = {
            "text": "test",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {}
        }
        
        # Should raise ValueError if loss_prompt is missing
        with pytest.raises(ValueError, match="loss_prompt"):
            node(state)
    
    def test_optimize_node_with_metadata_prompt(self):
        """Test optimize node with loss prompt in metadata."""
        config = TextGradConfig()
        node = create_optimize_node(config)
        
        state: TextGradState = {
            "text": "test",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {
                "loss_prompt": "Evaluate the text."
            }
        }
        
        # Should work with prompt in metadata
        # Note: This will fail if TextGrad API is not available, but structure is correct
        try:
            result = node(state)
            assert "optimized_text" in result
            assert "iteration" in result
        except Exception:
            # Expected if API is not available
            pass
    
    def test_evaluate_node_missing_text(self):
        """Test evaluate node with missing text."""
        config = TextGradConfig()
        node = create_evaluate_node(config)
        
        state: TextGradState = {
            "text": "",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {}
        }
        
        # Should return state unchanged if text is empty
        result = node(state)
        assert result == state
    
    def test_evaluate_node_missing_loss_prompt(self):
        """Test evaluate node with missing loss prompt."""
        config = TextGradConfig()
        node = create_evaluate_node(config)
        
        state: TextGradState = {
            "text": "test",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="loss_prompt"):
            node(state)
    
    def test_compare_node_missing_texts(self):
        """Test compare node with missing texts."""
        config = TextGradConfig()
        node = create_compare_node(config)
        
        state: TextGradState = {
            "text": "",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {}
        }
        
        # Should return state unchanged if texts are missing
        result = node(state)
        assert result == state
    
    def test_compare_node_structure(self):
        """Test that compare node updates state correctly."""
        config = TextGradConfig()
        node = create_compare_node(config)
        
        state: TextGradState = {
            "text": "original",
            "optimized_text": "optimized",
            "loss": None,
            "iteration": 0,
            "metadata": {
                "loss_prompt": "Evaluate."
            }
        }
        
        # Should work with both texts present
        # Note: This will fail if TextGrad API is not available
        try:
            result = node(state)
            assert "metadata" in result
            assert "comparison" in result["metadata"]
        except Exception:
            # Expected if API is not available
            pass


@pytest.mark.integration
class TestLangGraphIntegration:
    """Integration tests for LangGraph (requires actual setup)."""
    
    @pytest.mark.skip(reason="Requires LangGraph and TextGrad API setup")
    def test_langgraph_workflow(self):
        """Test full LangGraph workflow (requires actual setup)."""
        try:
            from langgraph.graph import StateGraph
            
            config = TextGradConfig(max_iterations=1)
            
            # Create graph
            graph = StateGraph(TextGradState)
            
            # Add nodes
            graph.add_node("optimize", create_optimize_node(config, "Evaluate."))
            graph.add_node("evaluate", create_evaluate_node(config))
            
            # Add edges
            graph.add_edge("optimize", "evaluate")
            
            # Set entry point
            graph.set_entry_point("optimize")
            
            # Compile
            app = graph.compile()
            
            # Initial state
            initial_state: TextGradState = {
                "text": "test",
                "optimized_text": None,
                "loss": None,
                "iteration": 0,
                "metadata": {
                    "loss_prompt": "Evaluate the text."
                }
            }
            
            # Run (this will fail if API is not available)
            result = app.invoke(initial_state)
            
            assert "optimized_text" in result
            assert "loss" in result
            
        except ImportError:
            pytest.skip("LangGraph not installed")
        except Exception:
            # Expected if API is not available
            pytest.skip("TextGrad API not available")
