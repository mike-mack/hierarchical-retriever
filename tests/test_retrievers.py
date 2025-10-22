"""
Tests for the retrievers module.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.retrievers import HierarchicalRetriever


class TestHierarchicalRetriever:
    """Test the HierarchicalRetriever class."""
    
    def test_initialization(self, mock_vector_store):
        """Test retriever initialization with default parameters."""
        retriever = HierarchicalRetriever(
            doc_store=mock_vector_store,
            chunk_store=mock_vector_store
        )
        
        assert retriever.doc_store == mock_vector_store
        assert retriever.chunk_store == mock_vector_store
        assert retriever.n_docs == 3
        assert retriever.n_chunks == 5
    
    def test_initialization_custom_params(self, mock_vector_store):
        """Test retriever initialization with custom parameters."""
        retriever = HierarchicalRetriever(
            doc_store=mock_vector_store,
            chunk_store=mock_vector_store,
            n_docs=5,
            n_chunks=10
        )
        
        assert retriever.n_docs == 5
        assert retriever.n_chunks == 10
    
    def test_get_relevant_documents_empty_results(self, mock_vector_store):
        """Test retrieval with no matching documents."""
        mock_vector_store.similarity_search.return_value = []
        
        retriever = HierarchicalRetriever(
            doc_store=mock_vector_store,
            chunk_store=mock_vector_store
        )
        
        results = retriever.get_relevant_documents("test query")
        
        assert results == []
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=3)
    
    def test_get_relevant_documents_with_results(self, mock_vector_store):
        """Test retrieval with matching documents and chunks."""
        # Setup coarse (document-level) results
        coarse_docs = [
            Document(
                page_content="Document 1 content",
                metadata={"source": "doc1.txt"}
            ),
            Document(
                page_content="Document 2 content",
                metadata={"source": "doc2.txt"}
            ),
        ]
        
        # Setup fine (chunk-level) results with scores
        chunk_results_doc1 = [
            (Document(page_content="Chunk 1 from doc1", metadata={"source": "doc1.txt"}), 0.95),
            (Document(page_content="Chunk 2 from doc1", metadata={"source": "doc1.txt"}), 0.90),
        ]
        
        chunk_results_doc2 = [
            (Document(page_content="Chunk 1 from doc2", metadata={"source": "doc2.txt"}), 0.85),
            (Document(page_content="Chunk 2 from doc2", metadata={"source": "doc2.txt"}), 0.80),
        ]
        
        # Configure mocks
        mock_doc_store = MagicMock()
        mock_chunk_store = MagicMock()
        
        mock_doc_store.similarity_search.return_value = coarse_docs
        mock_chunk_store.similarity_search_with_score.side_effect = [
            chunk_results_doc1,
            chunk_results_doc2
        ]
        
        retriever = HierarchicalRetriever(
            doc_store=mock_doc_store,
            chunk_store=mock_chunk_store,
            n_docs=2,
            n_chunks=2
        )
        
        results = retriever.get_relevant_documents("test query")
        
        # Assertions
        assert len(results) == 4  # 2 chunks from each of 2 documents
        mock_doc_store.similarity_search.assert_called_once_with("test query", k=2)
        assert mock_chunk_store.similarity_search_with_score.call_count == 2
        
        # Verify correct filters were used for each document
        calls = mock_chunk_store.similarity_search_with_score.call_args_list
        assert calls[0][0][0] == "test query"  # First call query
        assert calls[0][1]["k"] == 2  # First call k
        assert calls[0][1]["filter"] == {"source": "doc1.txt"}  # First call filter
        
        assert calls[1][0][0] == "test query"  # Second call query
        assert calls[1][1]["k"] == 2  # Second call k
        assert calls[1][1]["filter"] == {"source": "doc2.txt"}  # Second call filter
    
    def test_get_relevant_documents_respects_n_docs(self, mock_vector_store):
        """Test that n_docs parameter controls coarse retrieval."""
        mock_doc_store = MagicMock()
        mock_chunk_store = MagicMock()
        
        mock_doc_store.similarity_search.return_value = []
        
        retriever = HierarchicalRetriever(
            doc_store=mock_doc_store,
            chunk_store=mock_chunk_store,
            n_docs=7
        )
        
        retriever.get_relevant_documents("test query")
        
        mock_doc_store.similarity_search.assert_called_once_with("test query", k=7)
    
    def test_get_relevant_documents_respects_n_chunks(self, mock_vector_store):
        """Test that n_chunks parameter controls fine retrieval."""
        coarse_docs = [
            Document(page_content="Doc 1", metadata={"source": "doc1.txt"}),
        ]
        
        mock_doc_store = MagicMock()
        mock_chunk_store = MagicMock()
        
        mock_doc_store.similarity_search.return_value = coarse_docs
        mock_chunk_store.similarity_search_with_score.return_value = []
        
        retriever = HierarchicalRetriever(
            doc_store=mock_doc_store,
            chunk_store=mock_chunk_store,
            n_chunks=10
        )
        
        retriever.get_relevant_documents("test query")
        
        mock_chunk_store.similarity_search_with_score.assert_called_once_with(
            "test query",
            k=10,
            filter={"source": "doc1.txt"}
        )
    
    def test_get_relevant_documents_missing_source_metadata(self, mock_vector_store):
        """Test handling of documents without source metadata."""
        coarse_docs = [
            Document(page_content="Doc without source", metadata={}),
        ]
        
        mock_doc_store = MagicMock()
        mock_chunk_store = MagicMock()
        
        mock_doc_store.similarity_search.return_value = coarse_docs
        mock_chunk_store.similarity_search_with_score.return_value = []
        
        retriever = HierarchicalRetriever(
            doc_store=mock_doc_store,
            chunk_store=mock_chunk_store
        )
        
        results = retriever.get_relevant_documents("test query")
        
        # Should still attempt to search with None as source
        mock_chunk_store.similarity_search_with_score.assert_called_once_with(
            "test query",
            k=5,
            filter={"source": None}
        )
