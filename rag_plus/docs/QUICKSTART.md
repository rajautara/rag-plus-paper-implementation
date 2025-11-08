# RAG+ Quick Start Guide

Get started with RAG+ in 5 minutes!

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd rag-plus-paper-implementation
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up OpenAI API Key (Recommended)

```bash
# Linux/Mac
export OPENAI_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY='your-api-key-here'
```

Or set it in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

---

## Basic Usage

### Example 1: Simple Math Problem

```python
from rag_plus import RAGPlus, KnowledgeItem, OpenAILLM, OpenAIEmbeddingModel

# Initialize components
llm = OpenAILLM(model="gpt-3.5-turbo")
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
rag_plus = RAGPlus(llm, embedding_model, top_k=3)

# Define knowledge
knowledge_items = [
    KnowledgeItem(
        id="math_001",
        content="Power rule: d/dx(x^n) = n*x^(n-1)",
        knowledge_type="procedural",
        metadata={"category": "calculus"}
    )
]

# Build corpus (generates application examples)
applications = rag_plus.build_corpus(knowledge_items, use_generation=True)

# Ask a question
answer = rag_plus.generate(
    "What is the derivative of x^3?",
    task_type="math"
)
print(answer)
```

### Example 2: Without API Key (Using Mock)

```python
from rag_plus import RAGPlus, KnowledgeItem, SimpleEmbeddingModel

# Mock LLM for testing
class MockLLM:
    def generate(self, prompt, temperature=0.0, max_tokens=2048):
        return "Question: Sample?\nAnswer: Sample answer."

# Initialize with mock components
llm = MockLLM()
embedding_model = SimpleEmbeddingModel()
rag_plus = RAGPlus(llm, embedding_model, top_k=2)

# Rest is the same...
knowledge_items = [...]
applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
answer = rag_plus.generate("Your question?")
```

### Example 3: Save and Load Corpus

```python
# Build and save corpus
rag_plus = RAGPlus(llm, embedding_model)
applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
rag_plus.save_corpus("knowledge.json", "applications.json")

# Later, load corpus in new session
new_rag_plus = RAGPlus(llm, embedding_model)
new_rag_plus.load_corpus("knowledge.json", "applications.json")
answer = new_rag_plus.generate("Your question?")
```

---

## Configuration

### LLM Configuration

#### OpenAI Models

```python
# GPT-3.5 Turbo (fast, cost-effective)
llm = OpenAILLM(model="gpt-3.5-turbo")

# GPT-4 (more capable)
llm = OpenAILLM(model="gpt-4")

# GPT-4 Turbo (faster GPT-4)
llm = OpenAILLM(model="gpt-4-turbo")

# With custom parameters
llm = OpenAILLM(
    model="gpt-3.5-turbo",
    api_key="your-key",
    organization="your-org",
    base_url="https://custom-endpoint.com/v1"
)
```

### Embedding Model Configuration

```python
# Small model (1536 dimensions, cost-effective)
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")

# Large model (3072 dimensions, higher quality)
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-large")

# Ada-002 (older model)
embedding_model = OpenAIEmbeddingModel(model="text-embedding-ada-002")
```

### RAG+ Configuration

```python
# Configure number of retrieved results
rag_plus = RAGPlus(llm, embedding_model, top_k=5)  # Retrieve top 5

# Build corpus with different strategies
applications = rag_plus.build_corpus(
    knowledge_items,
    use_generation=True,   # Generate synthetic examples
    use_matching=False     # Match real-world cases
)
```

---

## Common Patterns

### Pattern 1: Domain-Specific Knowledge Base

```python
# Mathematics domain
math_knowledge = [
    KnowledgeItem(
        id="calc_001",
        content="Power rule for derivatives",
        knowledge_type="procedural",
        metadata={"category": "calculus", "difficulty": "basic"}
    ),
    KnowledgeItem(
        id="calc_002",
        content="Chain rule for derivatives",
        knowledge_type="procedural",
        metadata={"category": "calculus", "difficulty": "intermediate"}
    )
]

rag_plus = RAGPlus(llm, embedding_model)
rag_plus.build_corpus(math_knowledge, use_generation=True)

# Ask domain-specific questions
answer = rag_plus.generate(
    "Find derivative of sin(2x^2)",
    task_type="math"
)
```

### Pattern 2: Multiple Knowledge Types

```python
knowledge_items = [
    # Conceptual knowledge
    KnowledgeItem(
        id="concept_001",
        content="Pythagorean theorem: a² + b² = c²",
        knowledge_type="conceptual"
    ),
    # Procedural knowledge
    KnowledgeItem(
        id="proc_001",
        content="To solve quadratic: use formula x = [-b ± √(b²-4ac)]/2a",
        knowledge_type="procedural"
    )
]
```

### Pattern 3: Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "What is the derivative of x^2?",
    "How to integrate x^3?",
    "Solve equation: 2x + 5 = 11"
]

answers = []
for query in queries:
    answer = rag_plus.generate(query, task_type="math")
    answers.append(answer)

# Or use parallel processing
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    answers = list(executor.map(
        lambda q: rag_plus.generate(q, task_type="math"),
        queries
    ))
```

### Pattern 4: Incremental Corpus Building

```python
# Start with initial knowledge
initial_knowledge = [...]
rag_plus.build_corpus(initial_knowledge, use_generation=True)

# Save
rag_plus.save_corpus("v1_knowledge.json", "v1_apps.json")

# Later, load and add more
rag_plus.load_corpus("v1_knowledge.json", "v1_apps.json")

# Add new knowledge
new_knowledge = [...]
new_apps = rag_plus.build_corpus(new_knowledge, use_generation=True)

# Save updated version
rag_plus.save_corpus("v2_knowledge.json", "v2_apps.json")
```

### Pattern 5: Custom Metadata Filtering

```python
# Add rich metadata
knowledge_items = [
    KnowledgeItem(
        id="k1",
        content="...",
        knowledge_type="procedural",
        metadata={
            "category": "calculus",
            "difficulty": "easy",
            "tags": ["derivatives", "power-rule"],
            "prerequisites": ["algebra"]
        }
    )
]

# Use metadata for organization
def get_by_difficulty(retriever, difficulty):
    return [
        k for k in retriever.knowledge_corpus
        if k.metadata and k.metadata.get("difficulty") == difficulty
    ]

easy_items = get_by_difficulty(rag_plus.retriever, "easy")
```

---

## Troubleshooting

### Issue 1: Import Error for OpenAI

**Error:**
```
ImportError: OpenAI package not found
```

**Solution:**
```bash
pip install openai>=1.0.0
```

### Issue 2: API Key Not Found

**Error:**
```
openai.OpenAIError: No API key provided
```

**Solution:**
Set the API key as environment variable or pass explicitly:
```python
# Option 1: Environment variable
export OPENAI_API_KEY='your-key'

# Option 2: Pass to constructor
llm = OpenAILLM(api_key="your-key")
```

### Issue 3: Empty Retrieval Results

**Problem:** `rag_plus.generate()` returns baseline answer without retrieval.

**Solution:** Make sure corpus is built before generation:
```python
# Must build corpus first
applications = rag_plus.build_corpus(knowledge_items, use_generation=True)

# Then generate
answer = rag_plus.generate("query")
```

### Issue 4: Out of Memory

**Problem:** Large corpus causes memory issues.

**Solution:**
- Reduce `top_k` parameter
- Process in batches
- Use disk-based vector storage (FAISS)

```python
# Reduce retrieval size
rag_plus = RAGPlus(llm, embedding_model, top_k=3)  # Instead of 10

# Process in smaller batches
for batch in chunks(knowledge_items, batch_size=100):
    rag_plus.build_corpus(batch, use_generation=True)
```

### Issue 5: Slow Generation

**Problem:** Generation takes too long.

**Solution:**
- Use faster model (gpt-3.5-turbo instead of gpt-4)
- Reduce max_tokens
- Cache results

```python
# Use faster model
llm = OpenAILLM(model="gpt-3.5-turbo")

# Reduce token limit
answer = llm.generate(prompt, max_tokens=500)  # Instead of 2048

# Cache repeated queries
cache = {}
def cached_generate(query):
    if query not in cache:
        cache[query] = rag_plus.generate(query)
    return cache[query]
```

---

## Next Steps

### Learn More

- **[API Reference](API.md)**: Detailed API documentation
- **[Architecture](ARCHITECTURE.md)**: System design and components
- **[Deployment](DEPLOYMENT.md)**: Production deployment guide

### Try Examples

```bash
# Run example script
python example_usage.py

# Run domain-specific examples
python examples/math_examples.py
python examples/medical_examples.py
python examples/legal_examples.py
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_rag_plus.py

# Run with coverage
pytest --cov=rag_plus tests/
```

### Customize

- Implement custom LLM provider
- Add domain-specific prompts
- Create specialized retrievers
- Build evaluation metrics

---

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check all docs in `docs/` folder
- **Examples**: Review code in `examples/` folder
- **Tests**: Look at `tests/` for usage patterns

---

## Quick Reference

### Minimal Example

```python
from rag_plus import RAGPlus, KnowledgeItem, OpenAILLM, OpenAIEmbeddingModel

# Setup
llm = OpenAILLM()
embedding_model = OpenAIEmbeddingModel()
rag_plus = RAGPlus(llm, embedding_model)

# Build
knowledge = [KnowledgeItem(id="1", content="...", knowledge_type="procedural")]
rag_plus.build_corpus(knowledge, use_generation=True)

# Query
answer = rag_plus.generate("Your question?", task_type="general")
```

### Key Functions

```python
# Build corpus
rag_plus.build_corpus(knowledge_items, use_generation=True)

# Generate answer
rag_plus.generate(query, task_type="math")

# Save corpus
rag_plus.save_corpus("knowledge.json", "apps.json")

# Load corpus
rag_plus.load_corpus("knowledge.json", "apps.json")
```

---

Happy coding with RAG+! 🚀
