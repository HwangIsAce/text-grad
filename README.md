# TextGrad Module

A reusable Python module for TextGrad text optimization with LangGraph integration support.

## Overview

This module provides a clean, reusable interface for using [TextGrad](https://github.com/zou-group/textgrad) to optimize text and prompts. It includes:

- **TextGradOptimizer**: Wrapper for TextGrad optimization functionality
- **TextGradEvaluator**: Tools for evaluating and comparing texts
- **LangGraph Integration**: Ready-to-use node factories for LangGraph workflows
- **Configuration Management**: Flexible configuration with environment variable support

## Installation

```bash
# Install the module
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## API Key Setup

This module requires an OpenAI API key to use TextGrad. Set it up using one of the following methods:

### Option 1: Environment Variable (Recommended)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option 2: .env File

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key:
```
OPENAI_API_KEY=your-api-key-here
```

3. The module will automatically load the `.env` file if `python-dotenv` is installed.

**Note**: Make sure `.env` is in your `.gitignore` to avoid committing your API key.

## Quick Start

### Basic Text Optimization

```python
from text_grad_module import TextGradConfig, TextGradOptimizer
import textgrad as tg

# Configure
config = TextGradConfig(backward_engine="gpt-4o", max_iterations=1)
optimizer = TextGradOptimizer(config)

# Optimize text
initial_text = "This is a test sentence."
loss_fn = tg.TextLoss("Evaluate and improve this text.")
result = optimizer.optimize(initial_text, loss_fn)

print(result["optimized_text"])
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph
from text_grad_module import TextGradState, create_optimize_node, create_evaluate_node

# Create graph
graph = StateGraph(TextGradState)
graph.add_node("optimize", create_optimize_node(config, "Improve this text."))
graph.add_node("evaluate", create_evaluate_node(config))
graph.add_edge("optimize", "evaluate")

# Run
app = graph.compile()
result = app.invoke({"text": "Test", "optimized_text": None, "loss": None, "iteration": 0, "metadata": {}})
```

## Running Examples

```bash
# Basic usage examples
python examples/example_usage.py

# LangGraph integration examples
python examples/langgraph_example.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (skip integration tests)
pytest -m "not integration"

# Run only integration tests (requires API keys)
pytest -m integration
```

## Configuration

You can configure TextGrad using environment variables:

```bash
export TEXTGRAD_BACKWARD_ENGINE="gpt-4o"
export TEXTGRAD_MAX_ITERATIONS="1"
export TEXTGRAD_LEARNING_RATE="1.0"
```

Or programmatically:

```python
from text_grad_module import TextGradConfig

config = TextGradConfig(
    backward_engine="gpt-4o",
    max_iterations=1,
    learning_rate=1.0
)
```

## Requirements

- Python >= 3.12
- textgrad >= 0.1.6
- langgraph (optional, for LangGraph integration)
