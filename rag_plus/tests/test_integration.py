"""
Integration tests for RAG+ system.

Tests cover:
- End-to-end workflow
- Domain-specific examples
- OpenAI integration (if API key available)
- Corpus persistence
"""

import unittest
import os
import tempfile
import json
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_plus import (
    RAGPlus,
    KnowledgeItem,
    ApplicationExample,
    OpenAILLM,
    OpenAIEmbeddingModel,
    SimpleEmbeddingModel
)


class MockLLM:
    """Mock LLM for testing."""

    def generate(self, prompt, temperature=0.0, max_tokens=2048):
        """Generate mock response."""
        if "mathematical" in prompt.lower():
            return "Using the power rule: d/dx(x^3) = 3x^2"
        elif "medical" in prompt.lower():
            return "This appears to be Leriche syndrome based on the symptoms."
        elif "legal" in prompt.lower():
            return "Based on Article 234, the sentence should be A) Less than or equal to 36 months."
        else:
            return "Question: Sample question\nAnswer: Sample answer"


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete RAG+ workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.embedding_model = SimpleEmbeddingModel()

    def test_complete_math_workflow(self):
        """Test complete workflow for mathematics domain."""
        # Initialize RAG+
        rag_plus = RAGPlus(self.llm, self.embedding_model, top_k=2)

        # Create knowledge items
        knowledge_items = [
            KnowledgeItem(
                id="math_001",
                content="Power rule: d/dx(x^n) = n*x^(n-1)",
                knowledge_type="procedural",
                metadata={"category": "calculus", "domain": "mathematics"}
            ),
            KnowledgeItem(
                id="math_002",
                content="Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)",
                knowledge_type="procedural",
                metadata={"category": "calculus", "domain": "mathematics"}
            )
        ]

        # Build corpus
        applications = rag_plus.build_corpus(
            knowledge_items,
            use_generation=True,
            use_matching=False
        )

        self.assertEqual(len(applications), 2)

        # Generate answer
        query = "What is the derivative of x^3?"
        answer = rag_plus.generate(query, task_type="math")

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_complete_medical_workflow(self):
        """Test complete workflow for medical domain."""
        rag_plus = RAGPlus(self.llm, self.embedding_model, top_k=2)

        knowledge_items = [
            KnowledgeItem(
                id="med_001",
                content="Leriche syndrome: Triad of claudication, absent femoral pulses, and erectile dysfunction",
                knowledge_type="conceptual",
                metadata={"category": "vascular", "domain": "medical"}
            )
        ]

        applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
        self.assertEqual(len(applications), 1)

        query = "Patient with claudication, weak pulses, and erectile dysfunction?"
        answer = rag_plus.generate(query, task_type="medical")

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_complete_legal_workflow(self):
        """Test complete workflow for legal domain."""
        rag_plus = RAGPlus(self.llm, self.embedding_model, top_k=2)

        knowledge_items = [
            KnowledgeItem(
                id="law_001",
                content="Article 234: Intentional injury with minor injuries - up to 3 years imprisonment",
                knowledge_type="procedural",
                metadata={"category": "criminal_law", "domain": "legal"}
            )
        ]

        applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
        self.assertEqual(len(applications), 1)

        query = "Sentencing for first-degree minor injuries with voluntary surrender?"
        answer = rag_plus.generate(query, task_type="legal")

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


class TestCorpusPersistence(unittest.TestCase):
    """Test saving and loading corpus."""

    def test_save_and_load_workflow(self):
        """Test complete save/load workflow."""
        llm = MockLLM()
        embedding_model = SimpleEmbeddingModel()

        # Create and build initial system
        rag_plus = RAGPlus(llm, embedding_model, top_k=2)

        knowledge_items = [
            KnowledgeItem(
                id="test_001",
                content="Test knowledge content",
                knowledge_type="conceptual",
                metadata={"category": "test"}
            ),
            KnowledgeItem(
                id="test_002",
                content="Another test knowledge",
                knowledge_type="procedural",
                metadata={"category": "test"}
            )
        ]

        applications = rag_plus.build_corpus(knowledge_items, use_generation=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_path = os.path.join(tmpdir, "knowledge.json")
            applications_path = os.path.join(tmpdir, "applications.json")

            # Save
            rag_plus.save_corpus(knowledge_path, applications_path)

            # Verify files
            self.assertTrue(os.path.exists(knowledge_path))
            self.assertTrue(os.path.exists(applications_path))

            # Verify content
            with open(knowledge_path, 'r') as f:
                knowledge_data = json.load(f)
            self.assertEqual(len(knowledge_data), 2)

            with open(applications_path, 'r') as f:
                app_data = json.load(f)
            self.assertEqual(len(app_data), 2)

            # Load into new instance
            new_rag_plus = RAGPlus(llm, embedding_model, top_k=2)
            new_rag_plus.load_corpus(knowledge_path, applications_path)

            # Verify loaded data
            self.assertEqual(
                len(new_rag_plus.retriever.knowledge_corpus),
                len(knowledge_items)
            )
            self.assertEqual(
                len(new_rag_plus.retriever.application_corpus),
                len(applications)
            )

            # Test generation with loaded corpus
            answer = new_rag_plus.generate("Test query?")
            self.assertIsInstance(answer, str)


class TestMultipleRetrievalScenarios(unittest.TestCase):
    """Test different retrieval scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.embedding_model = SimpleEmbeddingModel()
        self.rag_plus = RAGPlus(self.llm, self.embedding_model, top_k=3)

    def test_retrieval_with_varied_top_k(self):
        """Test retrieval with different top_k values."""
        knowledge_items = [
            KnowledgeItem(id=f"k{i}", content=f"Knowledge {i}",
                         knowledge_type="conceptual")
            for i in range(10)
        ]

        # Test with top_k=1
        rag_plus_1 = RAGPlus(self.llm, self.embedding_model, top_k=1)
        rag_plus_1.build_corpus(knowledge_items, use_generation=True)
        results_1 = rag_plus_1.retriever.retrieve("test query")
        self.assertLessEqual(len(results_1), 1)

        # Test with top_k=5
        rag_plus_5 = RAGPlus(self.llm, self.embedding_model, top_k=5)
        rag_plus_5.build_corpus(knowledge_items, use_generation=True)
        results_5 = rag_plus_5.retriever.retrieve("test query")
        self.assertLessEqual(len(results_5), 5)

    def test_retrieval_with_no_indexed_corpus(self):
        """Test retrieval when no corpus is indexed."""
        # Don't build corpus
        query = "What is the answer?"
        answer = self.rag_plus.generate(query)

        # Should still generate an answer (fallback to baseline)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


@unittest.skipIf(
    not os.environ.get("OPENAI_API_KEY"),
    "Skipping OpenAI integration tests (no API key)"
)
class TestOpenAIIntegration(unittest.TestCase):
    """Test OpenAI integration (requires API key)."""

    def test_openai_llm_integration(self):
        """Test OpenAI LLM integration."""
        try:
            llm = OpenAILLM(model="gpt-3.5-turbo")
            embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")

            rag_plus = RAGPlus(llm, embedding_model, top_k=1)

            knowledge_items = [
                KnowledgeItem(
                    id="calc_001",
                    content="Power rule: d/dx(x^n) = n*x^(n-1)",
                    knowledge_type="procedural",
                    metadata={"category": "calculus"}
                )
            ]

            # Build corpus (this will call OpenAI API)
            applications = rag_plus.build_corpus(
                knowledge_items,
                use_generation=True,
                use_matching=False
            )

            self.assertEqual(len(applications), 1)
            self.assertIsInstance(applications[0].question, str)
            self.assertIsInstance(applications[0].answer, str)

            # Generate answer (this will call OpenAI API)
            answer = rag_plus.generate("What is the derivative of x^2?", task_type="math")

            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 10)  # Should have substantial content

        except Exception as e:
            self.fail(f"OpenAI integration test failed: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in integration scenarios."""

    def test_invalid_knowledge_type(self):
        """Test handling of invalid knowledge type."""
        llm = MockLLM()
        embedding_model = SimpleEmbeddingModel()
        rag_plus = RAGPlus(llm, embedding_model)

        # Create knowledge with invalid type
        knowledge_items = [
            KnowledgeItem(
                id="invalid_001",
                content="Test content",
                knowledge_type="invalid_type"  # Not 'conceptual' or 'procedural'
            )
        ]

        # Should handle error during generation
        applications = rag_plus.build_corpus(knowledge_items, use_generation=True)

        # Should not crash, but may have fewer applications
        self.assertIsInstance(applications, list)

    def test_corrupted_corpus_files(self):
        """Test loading corrupted corpus files."""
        llm = MockLLM()
        embedding_model = SimpleEmbeddingModel()
        rag_plus = RAGPlus(llm, embedding_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_path = os.path.join(tmpdir, "knowledge.json")
            applications_path = os.path.join(tmpdir, "applications.json")

            # Write invalid JSON
            with open(knowledge_path, 'w') as f:
                f.write("{ invalid json }")

            with open(applications_path, 'w') as f:
                f.write("{ invalid json }")

            # Should raise exception
            with self.assertRaises(json.JSONDecodeError):
                rag_plus.load_corpus(knowledge_path, applications_path)


if __name__ == "__main__":
    unittest.main()
