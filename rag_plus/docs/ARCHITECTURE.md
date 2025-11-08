# RAG+ Architecture Overview

This document provides a comprehensive overview of the RAG+ system architecture, design principles, and implementation details.

## Table of Contents
- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Principles](#design-principles)
- [Extensibility](#extensibility)

---

## System Overview

RAG+ (Retrieval-Augmented Generation Plus) extends traditional RAG systems by incorporating **application-aware reasoning** through a dual corpus approach:

```
┌─────────────────────────────────────────────────────────────┐
│                        RAG+ SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │  Knowledge   │◄────────┤ Application  │                │
│  │   Corpus     │         │   Corpus     │                │
│  └──────┬───────┘         └──────┬───────┘                │
│         │                        │                         │
│         │  Many-to-Many Links    │                         │
│         └────────────┬───────────┘                         │
│                      │                                     │
│              ┌───────▼────────┐                            │
│              │  Joint         │                            │
│              │  Retrieval     │                            │
│              └───────┬────────┘                            │
│                      │                                     │
│              ┌───────▼────────┐                            │
│              │  LLM           │                            │
│              │  Generation    │                            │
│              └────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Dual Corpus Architecture**: Separate storage for theoretical knowledge and practical applications
2. **Application-Aware Reasoning**: Retrieves both knowledge and examples of how to apply it
3. **Many-to-Many Linking**: Flexible associations between knowledge and applications
4. **Domain-Agnostic**: Supports mathematics, legal, medical, and other domains

---

## Core Components

### 1. Data Layer

#### KnowledgeItem
Represents theoretical knowledge or procedural rules.

```python
@dataclass
class KnowledgeItem:
    id: str                    # Unique identifier
    content: str               # Knowledge content
    knowledge_type: str        # 'conceptual' or 'procedural'
    metadata: Optional[Dict]   # Domain, category, etc.
```

**Types:**
- **Conceptual**: Definitions, theories, concepts (e.g., "What is anatomy?")
- **Procedural**: Steps, rules, formulas (e.g., "How to take derivative?")

#### ApplicationExample
Represents practical examples of knowledge application.

```python
@dataclass
class ApplicationExample:
    id: str                          # Unique identifier
    knowledge_id: str                # Linked knowledge
    question: str                    # Example problem
    answer: str                      # Solution
    reasoning_steps: Optional[List]  # Step-by-step reasoning
    metadata: Optional[Dict]         # Additional info
```

### 2. LLM Layer

Abstract interface allows swapping LLM providers:

```
┌─────────────────┐
│ LLMInterface    │ (Abstract)
└────────┬────────┘
         │
    ┌────┴─────┬──────────┐
    │          │          │
┌───▼────┐ ┌──▼───┐  ┌───▼────┐
│ OpenAI │ │ Mock │  │ Custom │
│  LLM   │ │ LLM  │  │  LLM   │
└────────┘ └──────┘  └────────┘
```

**Responsibilities:**
- Text generation
- Application creation
- Answer synthesis

### 3. Embedding Layer

Converts text to dense vectors for semantic search:

```
┌─────────────────────┐
│ EmbeddingModel      │ (Abstract)
└──────────┬──────────┘
           │
    ┌──────┴────┬────────────┐
    │           │            │
┌───▼──────┐ ┌─▼──────┐ ┌───▼───────┐
│ OpenAI   │ │ Simple │ │ Custom    │
│ Embedder │ │ Model  │ │ Embedder  │
└──────────┘ └────────┘ └───────────┘
```

**Responsibilities:**
- Text embedding
- Semantic similarity computation
- Vector normalization

### 4. Construction Pipeline

Builds application corpus from knowledge:

```
┌────────────────────────────────────────────┐
│   ApplicationCorpusConstructor             │
├────────────────────────────────────────────┤
│                                            │
│  Phase 1: Application Matching             │
│  ┌──────────────────────────────┐         │
│  │ • Category Alignment         │         │
│  │ • Semantic Matching          │         │
│  │ • Self-Consistency Voting    │         │
│  └──────────────────────────────┘         │
│                                            │
│  Phase 2: Application Generation           │
│  ┌──────────────────────────────┐         │
│  │ • LLM-based Generation       │         │
│  │ • Type-Specific Prompts      │         │
│  │ • Quality Control            │         │
│  └──────────────────────────────┘         │
└────────────────────────────────────────────┘
```

**Strategies:**

1. **Application Matching**: Link real-world cases to knowledge
   - Uses LLM with temperature sampling
   - Self-consistency voting (multiple runs)
   - Category-based alignment

2. **Application Generation**: Create synthetic examples
   - Conceptual knowledge → Comprehension questions
   - Procedural knowledge → Worked examples

### 5. Retrieval System

Retrieves relevant knowledge-application pairs:

```
┌────────────────────────────────────────────┐
│        RAGPlusRetriever                    │
├────────────────────────────────────────────┤
│                                            │
│  Input: Query                              │
│     ↓                                      │
│  Embed Query                               │
│     ↓                                      │
│  Similarity Search (Knowledge Corpus)      │
│     ↓                                      │
│  Retrieve Top-K Knowledge Items            │
│     ↓                                      │
│  Lookup Aligned Applications               │
│     ↓                                      │
│  Return Knowledge-Application Pairs        │
│                                            │
└────────────────────────────────────────────┘
```

**Features:**
- Vector-based similarity search
- Many-to-many knowledge-application links
- Configurable top-k retrieval
- Semantic fallback mechanism

### 6. Generation System

Main RAG+ orchestrator:

```
┌────────────────────────────────────────────┐
│             RAGPlus                        │
├────────────────────────────────────────────┤
│                                            │
│  build_corpus()                            │
│    ├─► Generate applications              │
│    ├─► Match applications                 │
│    └─► Index corpus                       │
│                                            │
│  generate(query)                           │
│    ├─► Retrieve pairs                     │
│    ├─► Create prompt                      │
│    └─► Generate answer                    │
│                                            │
│  save_corpus() / load_corpus()             │
│    └─► Persist/load data                  │
│                                            │
└────────────────────────────────────────────┘
```

---

## Data Flow

### Corpus Construction Flow

```
Knowledge Items
      │
      ├──► [Application Generation]
      │         │
      │         ├─► Create Conceptual Prompt
      │         ├─► Create Procedural Prompt
      │         └─► Parse LLM Response
      │              ↓
      │         Application Examples
      │              │
      ├──────────────┘
      │
      ├──► [Application Matching]
      │         │
      │         ├─► Categorize Items
      │         ├─► LLM Voting
      │         └─► Assign Links
      │              ↓
      │         Matched Applications
      │              │
      └──────────────┘
                     ↓
              [Index in Retriever]
                     ↓
            Knowledge + Applications
```

### Query Processing Flow

```
User Query
    │
    ├──► [Embed Query]
    │         ↓
    │    Query Vector
    │         │
    ├──► [Similarity Search]
    │         ↓
    │    Top-K Knowledge
    │         │
    ├──► [Lookup Applications]
    │         ↓
    │    Knowledge-Application Pairs
    │         │
    ├──► [Create RAG+ Prompt]
    │         │
    │         ├─► Format Knowledge
    │         ├─► Format Applications
    │         └─► Add Query
    │              ↓
    │         RAG+ Prompt
    │              │
    └──► [LLM Generation]
              ↓
         Final Answer
```

---

## Design Principles

### 1. Modularity

Each component has single responsibility:
- **LLM Layer**: Text generation only
- **Embedding Layer**: Vector representation only
- **Retrieval**: Finding relevant pairs only
- **Construction**: Building corpus only

### 2. Abstraction

Abstract interfaces enable flexibility:
```python
# Swap LLM providers easily
llm = OpenAILLM()  # or AnthropicLLM(), HuggingFaceLLM(), etc.
rag_plus = RAGPlus(llm, embedding_model)
```

### 3. Extensibility

Easy to extend for new domains:
```python
# Add new knowledge type
class ProceduralKnowledge(KnowledgeItem):
    steps: List[str]
    prerequisites: List[str]

# Add custom prompt strategy
def create_medical_prompt(knowledge, applications):
    # Domain-specific prompt logic
    pass
```

### 4. Persistence

Corpus can be saved/loaded:
```python
# Build once, use many times
rag_plus.build_corpus(knowledge_items)
rag_plus.save_corpus("knowledge.json", "apps.json")

# Later...
new_rag_plus.load_corpus("knowledge.json", "apps.json")
```

### 5. Type Safety

Strong typing with dataclasses and type hints:
```python
def retrieve(self, query: str) -> List[Tuple[KnowledgeItem, ApplicationExample]]:
    # Clear input/output types
    pass
```

---

## Extensibility

### Adding New LLM Provider

```python
class CustomLLM(LLMInterface):
    def __init__(self, api_key):
        self.client = CustomClient(api_key)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        response = self.client.complete(prompt, temp=temperature)
        return response.text

# Use it
llm = CustomLLM(api_key="...")
rag_plus = RAGPlus(llm, embedding_model)
```

### Adding New Embedding Model

```python
class CustomEmbeddingModel(EmbeddingModel):
    def __init__(self):
        self.model = load_custom_model()

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

# Use it
embedding_model = CustomEmbeddingModel()
rag_plus = RAGPlus(llm, embedding_model)
```

### Adding Domain-Specific Logic

```python
def create_domain_prompt(query, pairs, domain):
    if domain == "medicine":
        return f"Clinical reasoning for: {query}\n..."
    elif domain == "law":
        return f"Legal analysis for: {query}\n..."
    # ... more domains

# Extend RAGPlus
class DomainRAGPlus(RAGPlus):
    def _create_ragplus_prompt(self, query, pairs, task_type):
        return create_domain_prompt(query, pairs, task_type)
```

### Adding Reranking

```python
class RerankedRetriever(RAGPlusRetriever):
    def __init__(self, embedding_model, reranker, top_k=3):
        super().__init__(embedding_model, top_k)
        self.reranker = reranker

    def retrieve(self, query):
        # Initial retrieval
        candidates = super().retrieve(query)

        # Rerank
        reranked = self.reranker.rerank(query, candidates)
        return reranked[:self.top_k]
```

---

## Performance Considerations

### 1. Embedding Caching

```python
# Cache embeddings to avoid recomputation
embedding_cache = {}

def cached_embed(text):
    if text not in embedding_cache:
        embedding_cache[text] = embedding_model.embed([text])[0]
    return embedding_cache[text]
```

### 2. Batch Processing

```python
# Process multiple queries in batch
queries = ["query1", "query2", "query3"]
query_embeddings = embedding_model.embed(queries)  # Single API call
```

### 3. Vector Index Optimization

For large corpora, use FAISS or similar:
```python
import faiss

# Create FAISS index
dimension = 1536
index = faiss.IndexFlatIP(dimension)  # Inner product
index.add(knowledge_embeddings)

# Fast retrieval
D, I = index.search(query_embedding, k=top_k)
```

---

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies
- Verify edge cases

### Integration Tests
- Test end-to-end workflows
- Use real/mock LLMs
- Verify corpus persistence

### Domain Tests
- Test domain-specific examples
- Verify prompt generation
- Check answer quality

---

## See Also

- [API Reference](API.md)
- [Quick Start Guide](QUICKSTART.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Research Paper](../2506.11555v4.pdf)
