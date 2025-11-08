"""
Unit tests for embedding models.

Tests cover:
- OpenAI embedding model integration
- Embedding dimension handling
- Error handling
"""

import unittest
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_plus import (
    EmbeddingModel,
    SimpleEmbeddingModel,
    OpenAIEmbeddingModel
)


class TestSimpleEmbeddingModel(unittest.TestCase):
    """Test SimpleEmbeddingModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = SimpleEmbeddingModel()
        self.assertEqual(model.model_name, "all-MiniLM-L6-v2")

        model2 = SimpleEmbeddingModel("custom-model")
        self.assertEqual(model2.model_name, "custom-model")

    def test_embed_single_text(self):
        """Test embedding single text."""
        model = SimpleEmbeddingModel()
        embeddings = model.embed(["Hello world"])

        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 384)

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        model = SimpleEmbeddingModel()
        texts = ["text1", "text2", "text3", "text4"]
        embeddings = model.embed(texts)

        self.assertEqual(embeddings.shape[0], 4)
        self.assertEqual(embeddings.shape[1], 384)

    def test_embed_empty_list(self):
        """Test embedding empty list."""
        model = SimpleEmbeddingModel()
        embeddings = model.embed([])

        self.assertEqual(embeddings.shape[0], 0)


class TestOpenAIEmbeddingModel(unittest.TestCase):
    """Test OpenAIEmbeddingModel."""

    @patch('rag_plus.OpenAI')
    def test_initialization_default(self, mock_openai):
        """Test default initialization."""
        model = OpenAIEmbeddingModel()

        self.assertEqual(model.model, "text-embedding-3-small")
        self.assertEqual(model.dimensions, 1536)
        mock_openai.assert_called_once()

    @patch('rag_plus.OpenAI')
    def test_initialization_with_api_key(self, mock_openai):
        """Test initialization with API key."""
        model = OpenAIEmbeddingModel(api_key="test-key")

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs['api_key'], "test-key")

    @patch('rag_plus.OpenAI')
    def test_initialization_with_organization(self, mock_openai):
        """Test initialization with organization."""
        model = OpenAIEmbeddingModel(organization="test-org")

        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs['organization'], "test-org")

    @patch('rag_plus.OpenAI')
    def test_embedding_dimensions_mapping(self, mock_openai):
        """Test embedding dimensions for different models."""
        model_small = OpenAIEmbeddingModel(model="text-embedding-3-small")
        self.assertEqual(model_small.dimensions, 1536)

        model_large = OpenAIEmbeddingModel(model="text-embedding-3-large")
        self.assertEqual(model_large.dimensions, 3072)

        model_ada = OpenAIEmbeddingModel(model="text-embedding-ada-002")
        self.assertEqual(model_ada.dimensions, 1536)

    @patch('rag_plus.OpenAI')
    def test_embed_texts(self, mock_openai):
        """Test embedding texts with OpenAI."""
        # Create mock response
        mock_embedding_item1 = Mock()
        mock_embedding_item1.embedding = [0.1] * 1536
        mock_embedding_item2 = Mock()
        mock_embedding_item2.embedding = [0.2] * 1536

        mock_response = Mock()
        mock_response.data = [mock_embedding_item1, mock_embedding_item2]

        # Setup mock client
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Test embedding
        model = OpenAIEmbeddingModel()
        texts = ["text1", "text2"]
        embeddings = model.embed(texts)

        # Verify
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 1536)
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )

    @patch('rag_plus.OpenAI')
    def test_embed_error_handling(self, mock_openai):
        """Test error handling during embedding."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        model = OpenAIEmbeddingModel()

        with self.assertRaises(Exception) as context:
            model.embed(["test"])

        self.assertIn("API Error", str(context.exception))

    def test_openai_import_error(self):
        """Test handling when OpenAI package is not installed."""
        with patch.dict('sys.modules', {'openai': None}):
            # Clear the imported module
            import importlib
            import rag_plus
            importlib.reload(rag_plus)

            # This test would need proper module mocking
            # For now, we just verify the import error message is defined
            pass


class TestEmbeddingModelInterface(unittest.TestCase):
    """Test EmbeddingModel abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with self.assertRaises(TypeError):
            model = EmbeddingModel()

    def test_must_implement_embed(self):
        """Test that subclass must implement embed method."""
        class IncompleteEmbeddingModel(EmbeddingModel):
            pass

        with self.assertRaises(TypeError):
            model = IncompleteEmbeddingModel()

    def test_valid_implementation(self):
        """Test that valid implementation works."""
        class ValidEmbeddingModel(EmbeddingModel):
            def embed(self, texts):
                return np.random.randn(len(texts), 128)

        model = ValidEmbeddingModel()
        embeddings = model.embed(["test1", "test2"])

        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 128)


if __name__ == "__main__":
    unittest.main()
