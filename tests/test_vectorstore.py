"""
Tests for the vectorstore module.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.vectorstore import get_vector_store


class TestVectorStore:
    """Test vector store creation and configuration."""
    
    @patch('app.vectorstore.PGVector')
    @patch('app.vectorstore.OllamaEmbeddings')
    def test_get_vector_store_default_collection(self, mock_embeddings_class, mock_pgvector_class):
        """Test creating vector store with default collection name."""
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_store = MagicMock()
        mock_pgvector_class.return_value = mock_store
        
        result = get_vector_store()
        
        # Verify OllamaEmbeddings was created with correct parameters
        mock_embeddings_class.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Verify PGVector was created with correct parameters
        mock_pgvector_class.assert_called_once()
        call_kwargs = mock_pgvector_class.call_args[1]
        
        assert call_kwargs["embeddings"] == mock_embeddings
        assert call_kwargs["collection_name"] == "documents"
        assert call_kwargs["connection"] == "postgresql://test:test@localhost:5432/test_db"
        assert call_kwargs["use_jsonb"] is True
        
        assert result == mock_store
    
    @patch('app.vectorstore.PGVector')
    @patch('app.vectorstore.OllamaEmbeddings')
    def test_get_vector_store_custom_collection(self, mock_embeddings_class, mock_pgvector_class):
        """Test creating vector store with custom collection name."""
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_store = MagicMock()
        mock_pgvector_class.return_value = mock_store
        
        result = get_vector_store("custom_collection")
        
        # Verify custom collection name was used
        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"
    
    @patch('app.vectorstore.PGVector')
    @patch('app.vectorstore.OllamaEmbeddings')
    def test_get_vector_store_uses_env_vars(self, mock_embeddings_class, mock_pgvector_class, monkeypatch):
        """Test that environment variables are used for configuration."""
        # Set custom environment variables
        monkeypatch.setenv("DATABASE_URL", "postgresql://custom:pass@db:5432/custom_db")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-ollama:8080")
        
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_store = MagicMock()
        mock_pgvector_class.return_value = mock_store
        
        get_vector_store()
        
        # Verify custom Ollama URL was used
        mock_embeddings_class.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://custom-ollama:8080"
        )
        
        # Verify custom database URL was used
        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["connection"] == "postgresql://custom:pass@db:5432/custom_db"
    
    @patch('app.vectorstore.PGVector')
    @patch('app.vectorstore.OllamaEmbeddings')
    def test_get_vector_store_default_ollama_url(self, mock_embeddings_class, mock_pgvector_class, monkeypatch):
        """Test default Ollama URL when env var is not set."""
        # Remove OLLAMA_BASE_URL env var
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_store = MagicMock()
        mock_pgvector_class.return_value = mock_store
        
        get_vector_store()
        
        # Verify default Ollama URL was used
        mock_embeddings_class.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://host.docker.internal:11434"
        )
