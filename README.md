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

# Optional: Install GPU support for FAISS
pip install faiss-gpu
```

## Quick Start

```python
from rag_plus_core_implementation import RAGPlus, RAGPlusConfig, ApplicationExample

# Initialize RAG+ system with OpenAI embeddings
config = RAGPlusConfig(
    embedding_model="text-embedding-3-small",  # OpenAI embedding model
    embedding_type="openai",                # Use OpenAI embeddings
    openai_api_key="your-api-key-here"      # Your OpenAI API key
)
rag_plus = RAGPlus(config)

# Or use sentence-transformers embeddings
config_st = RAGPlusConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_type="sentence_transformer"
)
rag_plus_st = RAGPlus(config_st)

# Build corpora with full construction pipeline
existing_apps = [...]  # Optional existing applications
rag_plus.build_corpora("knowledge_source.txt", "mathematics", existing_apps)

# Generate response with enhanced retrieval
query = "How to solve this integration problem?"
response = rag_plus.generate_response(query, "mathematics", use_reranking=True)
print(response)

# Compare different retrieval methods
baseline_response = rag_plus.get_retrieval_agnostic_response(query, "mathematics", "knowledge_only")
apps_only_response = rag_plus.get_retrieval_agnostic_response(query, "mathematics", "applications_only")
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

### 1. Construction Pipeline
- `ApplicationMatcher`: Category alignment and semantic matching
- `ApplicationGenerator`: LLM-based application generation
- `Reranker`: Cross-encoder based reranking

### 2. Enhanced Corpus Management
- `KnowledgeCorpus`: Enhanced knowledge management
- `ApplicationCorpus`: Many-to-many link management
- `FAISSVectorStore`: Normalized vector storage with dynamic dimensions

### 3. RAG+ System
- `RAGPlus`: Main system with full pipeline
- Joint retrieval with semantic fallback
- Retrieval-agnostic modularity

### 4. Evaluation Framework
- `RAGPlusEvaluator`: Comprehensive evaluation system
- Multi-domain dataset loaders
- Method comparison and ablation studies

## Usage Examples

### Full Pipeline Usage

```python
from rag_plus_core_implementation import RAGPlus, RAGPlusConfig, ApplicationExample, KnowledgeType

# Configure system with OpenAI embeddings
config = RAGPlusConfig(
    embedding_model="text-embedding-3-small",  # OpenAI embedding model
    embedding_type="openai",                # Use OpenAI embeddings
    retrieval_top_k=3,
    application_top_k=2,
    llm_model="gpt-3.5-turbo",
    openai_api_key="your-api-key-here"      # Your OpenAI API key
)

# Or with sentence-transformers
config_st = RAGPlusConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_type="sentence_transformer",
    retrieval_top_k=3,
    application_top_k=2,
    llm_model="gpt-3.5-turbo"
)

# Initialize system with LLM client for generation
rag_plus = RAGPlus(config, llm_client)

# Build corpora with existing applications
existing_apps = [
    ApplicationExample(
        id="app_001",
        knowledge_id="calc_001",
        content="Worked example: Find derivative of x^3",
        question="What is the derivative of x^3?",
        answer="3x^2",
        application_type="worked_example"
    )
]

rag_plus.build_corpora("knowledge_source.txt", "mathematics", existing_apps)

# Generate response with reranking
response = rag_plus.generate_response(
    "What is the derivative of x^3?",
    "mathematics",
    use_reranking=True
)

# Compare retrieval methods
methods = ["rag_plus", "knowledge_only", "applications_only"]
for method in methods:
    response = rag_plus.get_retrieval_agnostic_response(
        "What is the derivative of x^3?",
        "mathematics",
        method
    )
    print(f"{method}: {response}")
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

## Configuration Options

```python
config = RAGPlusConfig(
    # Embedding settings
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    
    # Retrieval settings
    retrieval_top_k=3,
    application_top_k=2,
    
    # LLM settings
    llm_model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.0,
    
    # Domain settings
    domains=["mathematics", "legal", "medical"],
    
    # Performance settings
    cache_embeddings=True,
    batch_size=32
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
rag-plus-implementation/
├── rag_plus_core_implementation.py    # Core RAG+ implementation
├── rag_plus_evaluation.py             # Evaluation framework
├── rag_plus_integration.py            # Framework integrations
├── rag_plus_implementation_plan.md     # Detailed implementation plan
├── requirements.txt                   # Dependencies
├── README.md                         # This file
├── tests/                            # Test files
├── examples/                         # Usage examples
├── configs/                          # Configuration files
└── docs/                            # Documentation
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