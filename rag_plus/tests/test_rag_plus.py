"""
Unit tests for the RAG+ core implementation.

Tests cover:
- Knowledge and application data structures
- Application corpus construction
- Retrieval mechanisms
- RAG+ generation pipeline
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_plus import (
    KnowledgeItem,
    ApplicationExample,
    LLMInterface,
    EmbeddingModel,
    SimpleEmbeddingModel,
    ApplicationCorpusConstructor,
    RAGPlusRetriever,
    RAGPlus,
    compare_rag_vs_ragplus
)


class MockLLM(LLMInterface):
    """Mock LLM for testing."""

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """Return a mock response."""
        return "Question: Test question?\nAnswer: Test answer."


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing."""

    def embed(self, texts):
        """Return deterministic embeddings for testing."""
        # Simple hash-based embedding for reproducibility
        embeddings = []
        for text in texts:
            # Create a simple deterministic embedding
            hash_val = hash(text) % 1000
            embedding = np.ones(384) * hash_val / 1000.0
            embeddings.append(embedding)
        return np.array(embeddings)


class TestKnowledgeItem(unittest.TestCase):
    """Test KnowledgeItem data structure."""

    def test_knowledge_item_creation(self):
        """Test creating a knowledge item."""
        knowledge = KnowledgeItem(
            id="test_001",
            content="Test content",
            knowledge_type="conceptual",
            metadata={"category": "test"}
        )

        self.assertEqual(knowledge.id, "test_001")
        self.assertEqual(knowledge.content, "Test content")
        self.assertEqual(knowledge.knowledge_type, "conceptual")
        self.assertEqual(knowledge.metadata["category"], "test")

    def test_knowledge_item_without_metadata(self):
        """Test creating knowledge item without metadata."""
        knowledge = KnowledgeItem(
            id="test_002",
            content="Test content",
            knowledge_type="procedural"
        )

        self.assertEqual(knowledge.id, "test_002")
        self.assertIsNone(knowledge.metadata)


class TestApplicationExample(unittest.TestCase):
    """Test ApplicationExample data structure."""

    def test_application_creation(self):
        """Test creating an application example."""
        app = ApplicationExample(
            id="app_001",
            knowledge_id="test_001",
            question="What is this?",
            answer="This is a test.",
            reasoning_steps=["Step 1", "Step 2"]
        )

        self.assertEqual(app.id, "app_001")
        self.assertEqual(app.knowledge_id, "test_001")
        self.assertEqual(app.question, "What is this?")
        self.assertEqual(len(app.reasoning_steps), 2)


class TestApplicationCorpusConstructor(unittest.TestCase):
    """Test ApplicationCorpusConstructor."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.embedding_model = MockEmbeddingModel()
        self.constructor = ApplicationCorpusConstructor(self.llm, self.embedding_model)

    def test_generate_conceptual_application(self):
        """Test generating application for conceptual knowledge."""
        knowledge = KnowledgeItem(
            id="test_001",
            content="Pythagorean theorem: a² + b² = c²",
            knowledge_type="conceptual"
        )

        app = self.constructor.generate_application(knowledge)

        self.assertIsNotNone(app)
        self.assertEqual(app.knowledge_id, knowledge.id)
        self.assertIsInstance(app.question, str)
        self.assertIsInstance(app.answer, str)

    def test_generate_procedural_application(self):
        """Test generating application for procedural knowledge."""
        knowledge = KnowledgeItem(
            id="test_002",
            content="Power rule: d/dx(x^n) = n*x^(n-1)",
            knowledge_type="procedural"
        )

        app = self.constructor.generate_application(knowledge)

        self.assertIsNotNone(app)
        self.assertEqual(app.knowledge_id, knowledge.id)
        self.assertIsInstance(app.question, str)
        self.assertIsInstance(app.answer, str)

    def test_categorize_items(self):
        """Test categorization of knowledge items."""
        items = [
            KnowledgeItem(id="1", content="Math", knowledge_type="conceptual",
                         metadata={"category": "math"}),
            KnowledgeItem(id="2", content="Math2", knowledge_type="conceptual",
                         metadata={"category": "math"}),
            KnowledgeItem(id="3", content="Law", knowledge_type="conceptual",
                         metadata={"category": "law"}),
        ]

        categories = self.constructor._categorize_items(items)

        self.assertEqual(len(categories["math"]), 2)
        self.assertEqual(len(categories["law"]), 1)


class TestRAGPlusRetriever(unittest.TestCase):
    """Test RAGPlusRetriever."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding_model = MockEmbeddingModel()
        self.retriever = RAGPlusRetriever(self.embedding_model, top_k=2)

        # Create test knowledge items
        self.knowledge_items = [
            KnowledgeItem(
                id="k1",
                content="Python is a programming language",
                knowledge_type="conceptual"
            ),
            KnowledgeItem(
                id="k2",
                content="Machine learning is a subset of AI",
                knowledge_type="conceptual"
            ),
            KnowledgeItem(
                id="k3",
                content="Neural networks are used in deep learning",
                knowledge_type="conceptual"
            )
        ]

        # Create test applications
        self.applications = [
            ApplicationExample(
                id="app1",
                knowledge_id="k1",
                question="What is Python?",
                answer="Python is a programming language."
            ),
            ApplicationExample(
                id="app2",
                knowledge_id="k2",
                question="What is ML?",
                answer="Machine learning is a subset of AI."
            ),
            ApplicationExample(
                id="app3",
                knowledge_id="k3",
                question="What are neural networks?",
                answer="Neural networks are used in deep learning."
            )
        ]

    def test_index_knowledge(self):
        """Test indexing knowledge items."""
        self.retriever.index_knowledge(self.knowledge_items)

        self.assertEqual(len(self.retriever.knowledge_corpus), 3)
        self.assertIsNotNone(self.retriever.knowledge_embeddings)
        self.assertEqual(self.retriever.knowledge_embeddings.shape[0], 3)

    def test_index_applications(self):
        """Test indexing applications."""
        self.retriever.index_applications(self.applications)

        self.assertEqual(len(self.retriever.application_corpus), 3)
        self.assertEqual(len(self.retriever.knowledge_to_applications["k1"]), 1)
        self.assertEqual(len(self.retriever.knowledge_to_applications["k2"]), 1)

    def test_retrieve(self):
        """Test retrieving knowledge-application pairs."""
        self.retriever.index_knowledge(self.knowledge_items)
        self.retriever.index_applications(self.applications)

        query = "Tell me about programming languages"
        results = self.retriever.retrieve(query)

        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), self.retriever.top_k)

        if results:
            knowledge, application = results[0]
            self.assertIsInstance(knowledge, KnowledgeItem)
            self.assertIsInstance(application, ApplicationExample)
            self.assertEqual(knowledge.id, application.knowledge_id)


class TestRAGPlus(unittest.TestCase):
    """Test RAGPlus main system."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.embedding_model = MockEmbeddingModel()
        self.rag_plus = RAGPlus(self.llm, self.embedding_model, top_k=2)

        self.knowledge_items = [
            KnowledgeItem(
                id="k1",
                content="The derivative of x^n is n*x^(n-1)",
                knowledge_type="procedural",
                metadata={"category": "calculus"}
            ),
            KnowledgeItem(
                id="k2",
                content="Integration is the reverse of differentiation",
                knowledge_type="conceptual",
                metadata={"category": "calculus"}
            )
        ]

    def test_build_corpus_with_generation(self):
        """Test building corpus with generation."""
        applications = self.rag_plus.build_corpus(
            self.knowledge_items,
            use_generation=True,
            use_matching=False
        )

        self.assertEqual(len(applications), len(self.knowledge_items))
        self.assertEqual(len(self.rag_plus.retriever.knowledge_corpus), 2)
        self.assertEqual(len(self.rag_plus.retriever.application_corpus), 2)

    def test_generate_with_retrieval(self):
        """Test generating answer with retrieval."""
        self.rag_plus.build_corpus(self.knowledge_items, use_generation=True)

        query = "What is the derivative of x^3?"
        answer = self.rag_plus.generate(query, task_type="math")

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_generate_without_retrieval(self):
        """Test generating answer without any retrieved pairs."""
        # Don't build corpus, so no retrieval results
        query = "What is the meaning of life?"
        answer = self.rag_plus.generate(query)

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_save_and_load_corpus(self):
        """Test saving and loading corpus."""
        # Build corpus
        self.rag_plus.build_corpus(self.knowledge_items, use_generation=True)

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_path = os.path.join(tmpdir, "knowledge.json")
            applications_path = os.path.join(tmpdir, "applications.json")

            # Save corpus
            self.rag_plus.save_corpus(knowledge_path, applications_path)

            # Verify files exist
            self.assertTrue(os.path.exists(knowledge_path))
            self.assertTrue(os.path.exists(applications_path))

            # Load corpus in new instance
            new_rag_plus = RAGPlus(self.llm, self.embedding_model)
            new_rag_plus.load_corpus(knowledge_path, applications_path)

            # Verify loaded data
            self.assertEqual(
                len(new_rag_plus.retriever.knowledge_corpus),
                len(self.knowledge_items)
            )
            self.assertEqual(
                len(new_rag_plus.retriever.application_corpus),
                len(self.knowledge_items)
            )

    def test_prompt_creation_for_different_tasks(self):
        """Test prompt creation for different task types."""
        self.rag_plus.build_corpus(self.knowledge_items, use_generation=True)

        # Math task
        retrieved = self.rag_plus.retriever.retrieve("derivative")
        prompt_math = self.rag_plus._create_ragplus_prompt(
            "What is derivative?", retrieved, "math"
        )
        self.assertIn("mathematical problem", prompt_math)

        # Legal task
        prompt_legal = self.rag_plus._create_ragplus_prompt(
            "Legal question?", retrieved, "legal"
        )
        self.assertIn("legal question", prompt_legal)

        # Medical task
        prompt_medical = self.rag_plus._create_ragplus_prompt(
            "Medical question?", retrieved, "medical"
        )
        self.assertIn("medical question", prompt_medical)


class TestCompareRAGVsRAGPlus(unittest.TestCase):
    """Test comparison utilities."""

    def test_comparison(self):
        """Test comparing RAG+ with baseline."""
        llm = MockLLM()
        embedding_model = MockEmbeddingModel()
        rag_plus = RAGPlus(llm, embedding_model)

        knowledge_items = [
            KnowledgeItem(
                id="k1",
                content="Test knowledge",
                knowledge_type="conceptual"
            )
        ]

        rag_plus.build_corpus(knowledge_items, use_generation=True)

        test_queries = [
            {"query": "What is this?"},
            {"query": "How does this work?"}
        ]

        results = compare_rag_vs_ragplus(rag_plus, llm, test_queries)

        self.assertIn('baseline', results)
        self.assertIn('ragplus', results)
        self.assertIn('queries', results)
        self.assertEqual(len(results['baseline']), 2)
        self.assertEqual(len(results['ragplus']), 2)
        self.assertEqual(len(results['queries']), 2)


class TestSimpleEmbeddingModel(unittest.TestCase):
    """Test SimpleEmbeddingModel."""

    def test_simple_embedding_model(self):
        """Test simple embedding model generates embeddings."""
        model = SimpleEmbeddingModel()
        texts = ["text1", "text2", "text3"]
        embeddings = model.embed(texts)

        self.assertEqual(embeddings.shape[0], 3)
        self.assertEqual(embeddings.shape[1], 384)


if __name__ == "__main__":
    unittest.main()
