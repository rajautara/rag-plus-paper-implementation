# RAG+ API Reference

Complete API documentation for the RAG+ implementation.

## Table of Contents
- [Core Classes](#core-classes)
- [Data Structures](#data-structures)
- [LLM Interfaces](#llm-interfaces)
- [Embedding Models](#embedding-models)
- [Utilities](#utilities)

---

## Core Classes

### RAGPlus

Main class implementing the RAG+ system.

```python
class RAGPlus(llm, embedding_model, top_k=3)
```

**Parameters:**
- `llm` (LLMInterface): LLM interface for text generation
- `embedding_model` (EmbeddingModel): Embedding model for retrieval
- `top_k` (int, optional): Number of top results to retrieve. Default: 3

**Methods:**

#### `build_corpus(knowledge_items, real_world_cases=None, use_generation=True, use_matching=False)`

Build the application corpus aligned with knowledge corpus.

**Parameters:**
- `knowledge_items` (List[KnowledgeItem]): List of knowledge items
- `real_world_cases` (List[Dict], optional): Real-world cases for matching
- `use_generation` (bool): Whether to generate applications. Default: True
- `use_matching` (bool): Whether to match real-world cases. Default: False

**Returns:**
- `List[ApplicationExample]`: Generated/matched application examples

**Example:**
```python
rag_plus = RAGPlus(llm, embedding_model, top_k=3)

knowledge_items = [
    KnowledgeItem(
        id="math_001",
        content="Power rule: d/dx(x^n) = n*x^(n-1)",
        knowledge_type="procedural"
    )
]

applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
```

#### `generate(query, task_type="general")`

Generate answer for a query using RAG+ approach.

**Parameters:**
- `query` (str): Input query
- `task_type` (str): Type of task ("math", "legal", "medical", "general"). Default: "general"

**Returns:**
- `str`: Generated answer

**Example:**
```python
answer = rag_plus.generate(
    "What is the derivative of x^3?",
    task_type="math"
)
```

#### `save_corpus(knowledge_path, applications_path)`

Save knowledge and application corpus to JSON files.

**Parameters:**
- `knowledge_path` (str): Path to save knowledge corpus
- `applications_path` (str): Path to save applications corpus

**Example:**
```python
rag_plus.save_corpus("knowledge.json", "applications.json")
```

#### `load_corpus(knowledge_path, applications_path)`

Load knowledge and application corpus from JSON files.

**Parameters:**
- `knowledge_path` (str): Path to knowledge corpus file
- `applications_path` (str): Path to applications corpus file

**Example:**
```python
rag_plus.load_corpus("knowledge.json", "applications.json")
```

---

### ApplicationCorpusConstructor

Constructs application corpus aligned with knowledge corpus.

```python
class ApplicationCorpusConstructor(llm, embedding_model)
```

**Parameters:**
- `llm` (LLMInterface): LLM for generating applications
- `embedding_model` (EmbeddingModel): Model for embeddings

**Methods:**

#### `generate_application(knowledge)`

Generate an application example for a knowledge item.

**Parameters:**
- `knowledge` (KnowledgeItem): Knowledge item to generate application for

**Returns:**
- `ApplicationExample`: Generated application example

**Example:**
```python
constructor = ApplicationCorpusConstructor(llm, embedding_model)
knowledge = KnowledgeItem(
    id="k1",
    content="Pythagorean theorem",
    knowledge_type="conceptual"
)
app = constructor.generate_application(knowledge)
```

#### `match_applications(knowledge_items, real_world_cases, temperature=1.0, num_votes=3)`

Match real-world cases to knowledge items using LLM voting.

**Parameters:**
- `knowledge_items` (List[KnowledgeItem]): Knowledge items
- `real_world_cases` (List[Dict]): Real-world cases
- `temperature` (float): LLM temperature for sampling. Default: 1.0
- `num_votes` (int): Number of votes for consistency. Default: 3

**Returns:**
- `Dict[str, List[str]]`: Mapping of knowledge_id to case_ids

---

### RAGPlusRetriever

Retrieval mechanism for RAG+.

```python
class RAGPlusRetriever(embedding_model, top_k=3)
```

**Parameters:**
- `embedding_model` (EmbeddingModel): Embedding model
- `top_k` (int): Number of top results. Default: 3

**Methods:**

#### `index_knowledge(knowledge_items)`

Index knowledge items for retrieval.

**Parameters:**
- `knowledge_items` (List[KnowledgeItem]): Knowledge items to index

#### `index_applications(applications)`

Index application examples.

**Parameters:**
- `applications` (List[ApplicationExample]): Applications to index

#### `retrieve(query)`

Retrieve knowledge-application pairs for a query.

**Parameters:**
- `query` (str): Query string

**Returns:**
- `List[Tuple[KnowledgeItem, ApplicationExample]]`: Retrieved pairs

---

## Data Structures

### KnowledgeItem

Represents a knowledge item in the corpus.

```python
@dataclass
class KnowledgeItem:
    id: str
    content: str
    knowledge_type: str  # 'conceptual' or 'procedural'
    metadata: Optional[Dict] = None
```

**Example:**
```python
knowledge = KnowledgeItem(
    id="math_001",
    content="The power rule states: d/dx(x^n) = n*x^(n-1)",
    knowledge_type="procedural",
    metadata={"category": "calculus", "difficulty": "basic"}
)
```

### ApplicationExample

Represents an application example aligned with knowledge.

```python
@dataclass
class ApplicationExample:
    id: str
    knowledge_id: str
    question: str
    answer: str
    reasoning_steps: Optional[List[str]] = None
    metadata: Optional[Dict] = None
```

**Example:**
```python
application = ApplicationExample(
    id="app_001",
    knowledge_id="math_001",
    question="What is the derivative of x^3?",
    answer="Using the power rule: d/dx(x^3) = 3*x^2",
    reasoning_steps=["Apply power rule", "Simplify"],
    metadata={"difficulty": "easy"}
)
```

---

## LLM Interfaces

### LLMInterface (Abstract)

Abstract base class for LLM implementations.

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        pass
```

### OpenAILLM

OpenAI LLM implementation.

```python
class OpenAILLM(api_key=None, model="gpt-3.5-turbo", organization=None, base_url=None)
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key. If None, uses OPENAI_API_KEY env var
- `model` (str): Model name. Default: "gpt-3.5-turbo"
- `organization` (str, optional): OpenAI organization ID
- `base_url` (str, optional): Custom API endpoint

**Supported Models:**
- `gpt-3.5-turbo`: Cost-effective, fast
- `gpt-4`: More capable, higher quality
- `gpt-4-turbo`: Faster GPT-4 variant
- `gpt-4o`: Latest multimodal model

**Example:**
```python
# Using environment variable
llm = OpenAILLM(model="gpt-3.5-turbo")

# With explicit API key
llm = OpenAILLM(
    api_key="sk-...",
    model="gpt-4",
    organization="org-..."
)

# Generate text
response = llm.generate(
    "Explain the power rule",
    temperature=0.7,
    max_tokens=500
)
```

---

## Embedding Models

### EmbeddingModel (Abstract)

Abstract base class for embedding models.

```python
class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass
```

### OpenAIEmbeddingModel

OpenAI embedding implementation.

```python
class OpenAIEmbeddingModel(api_key=None, model="text-embedding-3-small", organization=None, base_url=None)
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key
- `model` (str): Embedding model name. Default: "text-embedding-3-small"
- `organization` (str, optional): OpenAI organization ID
- `base_url` (str, optional): Custom API endpoint

**Supported Models:**
- `text-embedding-3-small`: 1536 dimensions, cost-effective
- `text-embedding-3-large`: 3072 dimensions, higher quality
- `text-embedding-ada-002`: 1536 dimensions, older model

**Example:**
```python
embedding_model = OpenAIEmbeddingModel(
    model="text-embedding-3-small"
)

texts = ["Hello world", "Goodbye world"]
embeddings = embedding_model.embed(texts)
# Returns: numpy array of shape (2, 1536)
```

### SimpleEmbeddingModel

Placeholder embedding model for testing.

```python
class SimpleEmbeddingModel(model_name="all-MiniLM-L6-v2")
```

**Parameters:**
- `model_name` (str): Model name (placeholder). Default: "all-MiniLM-L6-v2"

**Note:** Returns random embeddings. For production, use OpenAIEmbeddingModel or implement actual model.

---

## Utilities

### `compare_rag_vs_ragplus(rag_system, baseline_llm, test_queries, task_type="general")`

Compare RAG+ performance against baseline.

**Parameters:**
- `rag_system` (RAGPlus): RAG+ system
- `baseline_llm` (LLMInterface): Baseline LLM
- `test_queries` (List[Dict]): Test queries with format `{"query": "..."}`
- `task_type` (str): Task type. Default: "general"

**Returns:**
- `Dict`: Comparison results with keys: "baseline", "ragplus", "queries"

**Example:**
```python
test_queries = [
    {"query": "What is the derivative of x^2?"},
    {"query": "How do you integrate x^3?"}
]

results = compare_rag_vs_ragplus(
    rag_plus,
    baseline_llm,
    test_queries,
    task_type="math"
)

print("Baseline answers:", results['baseline'])
print("RAG+ answers:", results['ragplus'])
```

---

## Error Handling

All classes may raise the following exceptions:

- `ImportError`: When required packages (openai) are not installed
- `ValueError`: For invalid parameters or configurations
- `FileNotFoundError`: When loading corpus files that don't exist
- `json.JSONDecodeError`: When loading malformed corpus files
- API exceptions from OpenAI SDK

**Example Error Handling:**
```python
try:
    llm = OpenAILLM(model="gpt-3.5-turbo")
    rag_plus = RAGPlus(llm, embedding_model)
    answer = rag_plus.generate("query")
except ImportError as e:
    print("OpenAI package not installed:", e)
except Exception as e:
    print("Error:", e)
```

---

## Type Hints

All public APIs include type hints for better IDE support:

```python
from typing import List, Dict, Tuple, Optional
from rag_plus import RAGPlus, KnowledgeItem, ApplicationExample

def process_knowledge(items: List[KnowledgeItem]) -> List[ApplicationExample]:
    rag_plus: RAGPlus = RAGPlus(llm, embedding_model)
    applications: List[ApplicationExample] = rag_plus.build_corpus(items)
    return applications
```

---

## Logging

RAG+ uses Python's `logging` module. Configure logging level:

```python
import logging

# Set to INFO for progress messages
logging.basicConfig(level=logging.INFO)

# Set to WARNING to reduce output
logging.basicConfig(level=logging.WARNING)

# Set to DEBUG for detailed information
logging.basicConfig(level=logging.DEBUG)
```

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [Quick Start Guide](QUICKSTART.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Main README](../README.md)
