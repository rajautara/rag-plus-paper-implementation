# RAG+ Implementation in Python

Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning

This repository provides a comprehensive Python implementation of the RAG+ framework as described in the research paper "RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning."

## Overview

RAG+ extends traditional RAG systems by incorporating application-aware reasoning through a dual corpus approach:
- **Knowledge Corpus**: Domain-specific facts, theorems, laws, medical knowledge
- **Application Corpus**: Practical examples demonstrating how to apply the knowledge
- **Joint Retrieval**: Simultaneous retrieval of knowledge and aligned applications

## Key Features

- 🧠 **Dual Corpus Architecture**: Separate storage for knowledge and application examples
- 🔗 **Knowledge-Application Mapping**: Intelligent pairing of knowledge with practical examples
- 🎯 **Domain-Specific Support**: Optimized for mathematics, legal, and medical domains
- 🔧 **Framework Integration**: Compatible with LangChain and LlamaIndex
- 📊 **Comprehensive Evaluation**: Built-in evaluation framework with multiple metrics
- 🚀 **REST API**: Ready-to-use API for deployment
- ⚡ **Performance Optimized**: Efficient vector storage and retrieval

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-plus-implementation

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key (required for OpenAI integration)
export OPENAI_API_KEY='your-api-key-here'

# Optional: Install GPU support for FAISS
pip install faiss-gpu
```

## Quick Start

### Using OpenAI SDK

```python
import os
from rag_plus import RAGPlus, KnowledgeItem, OpenAILLM, OpenAIEmbeddingModel

# Set your OpenAI API key (or export OPENAI_API_KEY environment variable)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize OpenAI components
llm = OpenAILLM(model="gpt-3.5-turbo")  # or "gpt-4", "gpt-4-turbo", etc.
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")

# Initialize RAG+ system
rag_plus = RAGPlus(llm, embedding_model, top_k=3)

# Define knowledge items
knowledge_items = [
    KnowledgeItem(
        id="math_001",
        content="The derivative of x^n is n*x^(n-1) according to the power rule.",
        knowledge_type="procedural",
        metadata={"category": "calculus"}
    )
]

# Build application corpus
applications = rag_plus.build_corpus(knowledge_items, use_generation=True)

# Generate answer
query = "What is the derivative of x^3?"
answer = rag_plus.generate(query, task_type="math")
print(answer)
```

### Alternative: Using Mock LLM (No API Key Required)

```python
from rag_plus import RAGPlus, KnowledgeItem, SimpleEmbeddingModel
from example_usage import MockLLM

# Use mock implementations for testing
llm = MockLLM()
embedding_model = SimpleEmbeddingModel()

# Initialize RAG+ system
rag_plus = RAGPlus(llm, embedding_model, top_k=3)

# Rest is the same as above...
```

## Architecture

```
RAG+ System:
├── Construction Pipeline
│   ├── Application Matching Phase
│   │   ├── Category Alignment
│   │   ├── Semantic Matching
│   │   └── Manual Refinement
│   └── Application Generation Phase
│       ├── Backfill for Missing Pairs
│       └── Type-Specific Generation
├── Enhanced Retrieval System
│   ├── Joint Knowledge-Application Retrieval
│   ├── Semantic Fallback Mechanism
│   └── Cross-Encoder Reranking
├── Many-to-Many Link Management
│   ├── Knowledge → Applications
│   └── Applications → Knowledge
└── Evaluation Framework
    ├── Multi-Domain Testing
    ├── Method Comparison
    └── Ablation Studies
```

## Core Components

### 1. LLM Interfaces
- `LLMInterface`: Abstract base class for LLM implementations
- `OpenAILLM`: OpenAI GPT integration (gpt-3.5-turbo, gpt-4, etc.)
- `MockLLM`: Mock implementation for testing without API keys

### 2. Embedding Models
- `EmbeddingModel`: Abstract base class for embedding implementations
- `OpenAIEmbeddingModel`: OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
- `SimpleEmbeddingModel`: Placeholder for local embedding models

### 3. Construction Pipeline
- `ApplicationCorpusConstructor`: Builds application examples from knowledge
- Category alignment and semantic matching
- LLM-based application generation

### 4. Retrieval System
- `RAGPlusRetriever`: Joint knowledge-application retrieval
- Vector similarity search using embeddings
- Top-k retrieval with configurable parameters

### 5. Main System
- `RAGPlus`: Complete RAG+ system
- Corpus building and indexing
- Query generation with application-aware reasoning

## Usage Examples

### Basic Usage with OpenAI

```python
import os
from rag_plus import (
    RAGPlus,
    KnowledgeItem,
    OpenAILLM,
    OpenAIEmbeddingModel
)

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize components
llm = OpenAILLM(model="gpt-3.5-turbo")
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
rag_plus = RAGPlus(llm, embedding_model, top_k=3)

# Create knowledge items
knowledge_items = [
    KnowledgeItem(
        id="math_001",
        content="Pythagorean theorem: a² + b² = c² for right triangles",
        knowledge_type="conceptual",
        metadata={"category": "geometry"}
    ),
    KnowledgeItem(
        id="math_002",
        content="Power rule: d/dx(x^n) = n*x^(n-1)",
        knowledge_type="procedural",
        metadata={"category": "calculus"}
    )
]

# Build corpus
applications = rag_plus.build_corpus(
    knowledge_items,
    use_generation=True,
    use_matching=False
)

# Generate answer
answer = rag_plus.generate("What is the derivative of x^5?", task_type="math")
print(answer)

# Save corpus for later use
rag_plus.save_corpus("knowledge.json", "applications.json")

# Load corpus in a new session
rag_plus_new = RAGPlus(llm, embedding_model)
rag_plus_new.load_corpus("knowledge.json", "applications.json")
```

### Advanced: Custom OpenAI Configuration

```python
from rag_plus import OpenAILLM, OpenAIEmbeddingModel

# With explicit API key and organization
llm = OpenAILLM(
    api_key="your-api-key",
    model="gpt-4",
    organization="your-org-id"
)

# Using different embedding models
embedding_small = OpenAIEmbeddingModel(model="text-embedding-3-small")  # 1536 dims
embedding_large = OpenAIEmbeddingModel(model="text-embedding-3-large")  # 3072 dims
embedding_ada = OpenAIEmbeddingModel(model="text-embedding-ada-002")    # 1536 dims

# Custom base URL (for Azure OpenAI or custom endpoints)
llm_custom = OpenAILLM(
    model="gpt-3.5-turbo",
    base_url="https://your-custom-endpoint.com/v1"
)
```

### Running the Example Script

```bash
# With OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
python example_usage.py

# Without API key (uses mock LLM)
python example_usage.py
```

### Framework Integration

```python
from rag_plus_integration import RAGPlusFrameworkAdapter

# Create adapter
adapter = RAGPlusFrameworkAdapter()

# Integrate with LangChain
langchain_rag_plus = adapter.create_integration("langchain", rag_plus)

# Integrate with LlamaIndex
llamaindex_rag_plus = adapter.create_integration("llamaindex", rag_plus)
```

### Comprehensive Evaluation

```python
from rag_plus_evaluation import RAGPlusEvaluator, DatasetLoader, create_baseline_methods

# Create evaluator
evaluator = RAGPlusEvaluator()

# Load datasets
datasets = [
    DatasetLoader.load_mathqa_dataset("mathqa.json"),
    DatasetLoader.load_legal_dataset("legal.json"),
    DatasetLoader.load_medical_dataset("medical.json")
]

# Compare different methods
methods = create_baseline_methods(config)
comparison_results = evaluator.compare_methods(methods, datasets)

# Generate comprehensive report with visualizations
evaluator.plot_results(comparison_results, "results.png")

# Print method comparison summary
print("\n=== Method Comparison ===")
for method in methods.keys():
    if methods[method] is not None:
        method_results = comparison_results[comparison_results['Method'] == method]
        print(f"\n{method}:")
        for metric in ['accuracy', 'reasoning_quality', 'response_time']:
            metric_results = method_results[method_results['Metric'] == metric]
            if not metric_results.empty:
                avg_value = metric_results['Value'].mean()
                print(f"  {metric}: {avg_value:.4f}")
```

### REST API

```python
from rag_plus_integration import RAGPlusAPI

# Create API
api = RAGPlusAPI(rag_plus)

# Run server (in production)
api.run(host="0.0.0.0", port=5000)

# Or use client code
import requests

response = requests.post("http://localhost:5000/query", json={
    "query": "How to solve this integration problem?",
    "domain": "mathematics"
})
print(response.json())
```

## Configuration

### Environment Variables

Set your OpenAI API key as an environment variable:

```bash
# Linux/Mac
export OPENAI_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY='your-api-key-here'
```

Or set it programmatically in Python:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### OpenAI Models

**LLM Models (for text generation):**
- `gpt-3.5-turbo` (default, cost-effective)
- `gpt-4` (more capable, higher cost)
- `gpt-4-turbo` (faster GPT-4)
- `gpt-4o` (latest multimodal model)

**Embedding Models:**
- `text-embedding-3-small` (default, 1536 dimensions, cost-effective)
- `text-embedding-3-large` (3072 dimensions, higher quality)
- `text-embedding-ada-002` (1536 dimensions, older model)

### RAG+ Configuration

```python
from rag_plus import RAGPlus, OpenAILLM, OpenAIEmbeddingModel

# Configure LLM
llm = OpenAILLM(
    model="gpt-3.5-turbo",  # or gpt-4, gpt-4-turbo, etc.
    api_key="your-api-key",  # optional if using env var
    organization="your-org"   # optional
)

# Configure embeddings
embedding_model = OpenAIEmbeddingModel(
    model="text-embedding-3-small",  # or text-embedding-3-large
    api_key="your-api-key"           # optional if using env var
)

# Configure RAG+ system
rag_plus = RAGPlus(
    llm=llm,
    embedding_model=embedding_model,
    top_k=3  # number of knowledge-application pairs to retrieve
)
```

## Supported Domains

### Mathematics
- Calculus, algebra, statistics
- Procedural problem-solving
- Step-by-step reasoning

### Legal
- Case analysis and sentencing
- Legal reasoning and interpretation
- Statute application

### Medical
- Clinical diagnosis
- Medical reasoning
- Treatment planning

## Evaluation Metrics

- **Accuracy**: Correctness of final answers
- **Reasoning Quality**: Quality of step-by-step reasoning
- **Response Time**: Performance metrics
- **Domain-Specific Metrics**: Custom evaluation per domain

## Performance Optimization

### Vector Storage
- FAISS for efficient similarity search
- Configurable embedding dimensions
- Batch processing support

### Caching
- Embedding cache for repeated queries
- Application cache for faster retrieval
- Configurable cache sizes

### Scalability
- Distributed processing support
- GPU acceleration for embeddings
- Asynchronous retrieval pipeline

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag_plus

# Run specific test file
pytest test_rag_plus.py
```

### Code Quality

```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Type checking
mypy rag_plus_core_implementation.py
```

## Project Structure

```
rag-plus-paper-implementation/
├── rag_plus.py                       # Core RAG+ implementation
│   ├── KnowledgeItem                 # Knowledge corpus data structure
│   ├── ApplicationExample            # Application corpus data structure
│   ├── LLMInterface                  # Abstract LLM interface
│   ├── OpenAILLM                     # OpenAI LLM implementation
│   ├── EmbeddingModel                # Abstract embedding interface
│   ├── OpenAIEmbeddingModel          # OpenAI embedding implementation
│   ├── SimpleEmbeddingModel          # Placeholder embedding model
│   ├── ApplicationCorpusConstructor  # Application generation & matching
│   ├── RAGPlusRetriever              # Knowledge-application retrieval
│   └── RAGPlus                       # Main RAG+ system
├── example_usage.py                  # Comprehensive usage examples
│   ├── MockLLM                       # Mock LLM for testing
│   ├── mathematics_example()         # Math domain demo
│   ├── medical_example()             # Medical domain demo
│   ├── legal_example()               # Legal domain demo
│   └── openai_example()              # OpenAI integration demo
├── requirements.txt                  # Python dependencies
├── IMPLEMENTATION_SUMMARY.md         # Implementation details
├── 2506.11555v4.pdf                  # Original research paper
├── README.md                         # This file
├── tests/                            # Unit and integration tests
│   ├── test_rag_plus.py             # Core functionality tests
│   ├── test_embeddings.py           # Embedding model tests
│   └── test_integration.py          # Integration tests
├── examples/                         # Domain-specific examples
│   ├── math_examples.py             # Mathematics domain examples
│   ├── medical_examples.py          # Medical domain examples
│   └── legal_examples.py            # Legal domain examples
└── docs/                            # Documentation
    ├── API.md                       # API reference
    ├── ARCHITECTURE.md              # Architecture overview
    ├── QUICKSTART.md                # Quick start guide
    └── DEPLOYMENT.md                # Deployment guide
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Research Paper Alignment

This implementation addresses all key aspects mentioned in the review:

### ✅ Implemented Features
- [x] **Full construction pipeline** (Matching + Generation)
- [x] **Many-to-many knowledge-application links**
- [x] **Semantic fallback for application retrieval**
- [x] **Cross-encoder reranking**
- [x] **Vector normalization for IP similarity**
- [x] **Dynamic embedding dimensions**
- [x] **Multi-domain evaluation harness**
- [x] **Retrieval-agnostic modularity**
- [x] **Application-aware retrieval improvements**

### 🔧 Technical Improvements
- [x] **FAISS vector normalization** for stable IP similarity
- [x] **Model-agnostic embedding dimensions**
- [x] **Category alignment in matching**
- [x] **Manual refinement simulation**
- [x] **Joint retrieval with fallback**
- [x] **Comprehensive evaluation metrics**

### 📊 Evaluation Coverage
- [x] **Multiple domains**: Mathematics, Legal, Medical
- [x] **Method comparison**: Baseline, RAG, RAG+, ablations
- [x] **Performance metrics**: Accuracy, reasoning quality, response time
- [x] **Visualization**: Comprehensive result plotting

### 🚀 Key Improvements Over Basic Implementation
1. **Construction Pipeline**: Implements both matching and generation phases
2. **Enhanced Retrieval**: Semantic fallback when direct links are insufficient
3. **Reranking**: Cross-encoder for improved relevance
4. **Many-to-Many Links**: Flexible knowledge-application associations
5. **Evaluation Framework**: Comprehensive comparison across methods and domains

## Citation

If you use this implementation in your research, please cite the original RAG+ paper:

```bibtex
@article{wang2025rag,
  title={RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning},
  author={Wang, Yu and Zhao, Shiwan and Wang, Zhihu and others},
  journal={arXiv preprint arXiv:2506.11555v4},
  year={2025}
}
```

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](issues)
- 💬 [Discussions](discussions)
- 📧 [Email Support](mailto:support@example.com)

## Acknowledgments

- Original RAG+ research paper authors
- Open-source RAG frameworks (LangChain, LlamaIndex)
- FAISS for efficient vector storage
- Hugging Face for embedding models