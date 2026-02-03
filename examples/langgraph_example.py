"""
LangGraph integration examples for TextGrad module.
"""

from text_grad_module import (
    TextGradConfig,
    TextGradState,
    create_optimize_node,
    create_evaluate_node,
    create_compare_node,
)


def example_langgraph_basic():
    """Example: Basic LangGraph workflow with TextGrad."""
    print("=== LangGraph Basic Workflow ===")
    
    try:
        from langgraph.graph import StateGraph
        
        # Configure TextGrad
        config = TextGradConfig(max_iterations=1)
        
        # Create graph
        graph = StateGraph(TextGradState)
        
        # Add nodes
        graph.add_node(
            "optimize",
            create_optimize_node(
                config,
                loss_prompt="Evaluate and improve this text for clarity and conciseness."
            )
        )
        graph.add_node("evaluate", create_evaluate_node(config))
        
        # Add edges
        graph.add_edge("optimize", "evaluate")
        
        # Set entry point
        graph.set_entry_point("optimize")
        
        # Compile
        app = graph.compile()
        
        # Initial state
        initial_state: TextGradState = {
            "text": "This is a test sentence.",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {
                "loss_prompt": "Evaluate the quality of this text."
            }
        }
        
        # Run
        result = app.invoke(initial_state)
        
        print(f"Original: {result['text']}")
        print(f"Optimized: {result.get('optimized_text', 'N/A')}")
        print(f"Loss: {result.get('loss', 'N/A')}")
        print()
        
    except ImportError:
        print("LangGraph is not installed. Install it with: pip install langgraph")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your API keys before running.")


def example_langgraph_with_compare():
    """Example: LangGraph workflow with comparison node."""
    print("=== LangGraph with Comparison ===")
    
    try:
        from langgraph.graph import StateGraph
        
        config = TextGradConfig(max_iterations=1)
        
        # Create graph
        graph = StateGraph(TextGradState)
        
        # Add nodes
        graph.add_node(
            "optimize",
            create_optimize_node(
                config,
                loss_prompt="Improve this text."
            )
        )
        graph.add_node("compare", create_compare_node(config))
        
        # Add edges
        graph.add_edge("optimize", "compare")
        
        # Set entry point
        graph.set_entry_point("optimize")
        
        # Compile
        app = graph.compile()
        
        # Initial state
        initial_state: TextGradState = {
            "text": "Bad text.",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {
                "loss_prompt": "Rate the quality from 0 to 10."
            }
        }
        
        # Run
        result = app.invoke(initial_state)
        
        comparison = result.get("metadata", {}).get("comparison", {})
        if comparison:
            print(f"Original Loss: {comparison.get('original_loss', 'N/A')}")
            print(f"Optimized Loss: {comparison.get('optimized_loss', 'N/A')}")
            print(f"Improved: {comparison.get('improvement', 'N/A')}")
        print()
        
    except ImportError:
        print("LangGraph is not installed. Install it with: pip install langgraph")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your API keys before running.")


def example_langgraph_conditional():
    """Example: LangGraph with conditional routing."""
    print("=== LangGraph Conditional Routing ===")
    
    try:
        from langgraph.graph import StateGraph
        
        config = TextGradConfig(max_iterations=1)
        
        # Create graph
        graph = StateGraph(TextGradState)
        
        # Add nodes
        graph.add_node(
            "optimize",
            create_optimize_node(
                config,
                loss_prompt="Improve this text."
            )
        )
        graph.add_node("evaluate", create_evaluate_node(config))
        
        # Conditional edge based on loss
        def should_continue(state: TextGradState) -> str:
            loss = state.get("loss")
            if loss is None:
                return "evaluate"
            # Continue if loss is high (example logic)
            try:
                loss_value = float(loss)
                return "optimize" if loss_value > 5 else "end"
            except (ValueError, TypeError):
                return "end"
        
        graph.add_edge("optimize", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            should_continue,
            {
                "optimize": "optimize",
                "end": "__end__"
            }
        )
        
        graph.set_entry_point("optimize")
        
        # Compile
        app = graph.compile()
        
        # Initial state
        initial_state: TextGradState = {
            "text": "This text needs improvement.",
            "optimized_text": None,
            "loss": None,
            "iteration": 0,
            "metadata": {
                "loss_prompt": "Rate the quality from 0 to 10."
            }
        }
        
        # Run
        result = app.invoke(initial_state)
        
        print(f"Final optimized text: {result.get('optimized_text', 'N/A')}")
        print(f"Final loss: {result.get('loss', 'N/A')}")
        print()
        
    except ImportError:
        print("LangGraph is not installed. Install it with: pip install langgraph")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your API keys before running.")


if __name__ == "__main__":
    print("TextGrad Module - LangGraph Integration Examples\n")
    
    # Uncomment the examples you want to run
    # Note: These require API keys and may incur costs
    
    # example_langgraph_basic()
    # example_langgraph_with_compare()
    # example_langgraph_conditional()
    
    print("Uncomment examples in the code to run them.")
    print("Make sure to set your API keys and install LangGraph before running.")
