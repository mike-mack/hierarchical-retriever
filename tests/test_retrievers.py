"""
Tests for the retrievers module.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.retrievers import HierarchicalRetriever, MetadataHierarchicalRetriever


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


class TestMetadataHierarchicalRetriever:
    """Test the MetadataHierarchicalRetriever class."""
    
    def test_initialization(self, mock_vector_store):
        """Test retriever initialization with default parameters."""
        retriever = MetadataHierarchicalRetriever(vector_store=mock_vector_store)
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.n_docs == 3
        assert retriever.n_chunks_per_doc == 5
    
    def test_initialization_custom_params(self, mock_vector_store):
        """Test retriever initialization with custom parameters."""
        retriever = MetadataHierarchicalRetriever(
            vector_store=mock_vector_store,
            n_docs=5,
            n_chunks_per_doc=10
        )
        
        assert retriever.n_docs == 5
        assert retriever.n_chunks_per_doc == 10
    
    def test_get_relevant_documents_empty_results(self, mock_vector_store):
        """Test retrieval with no matching summaries."""
        mock_vector_store.similarity_search_with_score.return_value = []
        
        retriever = MetadataHierarchicalRetriever(vector_store=mock_vector_store)
        results = retriever.get_relevant_documents("test query")
        
        assert results == []
        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            "test query",
            k=3,
            filter={"type": "summary"}
        )
    
    def test_get_relevant_documents_with_results(self, mock_vector_store):
        """Test retrieval with matching summaries and chunks."""
        # Mock summaries
        summaries = [
            (Document(
                page_content="Summary 1",
                metadata={"source": "doc1.txt", "type": "summary"}
            ), 0.1),
            (Document(
                page_content="Summary 2",
                metadata={"source": "doc2.txt", "type": "summary"}
            ), 0.2)
        ]
        
        # Mock chunks
        chunks_doc1 = [
            (Document(
                page_content="Chunk 1",
                metadata={"source": "doc1.txt", "type": "chunk", "chunk_index": 0}
            ), 0.05),
            (Document(
                page_content="Chunk 2",
                metadata={"source": "doc1.txt", "type": "chunk", "chunk_index": 1}
            ), 0.08)
        ]
        
        chunks_doc2 = [
            (Document(
                page_content="Chunk 3",
                metadata={"source": "doc2.txt", "type": "chunk", "chunk_index": 0}
            ), 0.06)
        ]
        
        # Setup mock to return different results based on filter
        def side_effect(query, k, filter):
            if filter.get("type") == "summary":
                return summaries[:k]
            elif filter.get("source") == "doc1.txt":
                return chunks_doc1[:k]
            elif filter.get("source") == "doc2.txt":
                return chunks_doc2[:k]
            return []
        
        mock_vector_store.similarity_search_with_score.side_effect = side_effect
        
        retriever = MetadataHierarchicalRetriever(
            vector_store=mock_vector_store,
            n_docs=2,
            n_chunks_per_doc=2
        )
        results = retriever.get_relevant_documents("test query")
        
        # Should return chunks from both documents
        assert len(results) == 3  # 2 from doc1, 1 from doc2
        
        # Verify parent_summary_score was added
        for chunk, score in results:
            assert "parent_summary_score" in chunk.metadata
    
    def test_get_relevant_documents_respects_n_docs(self, mock_vector_store):
        """Test that n_docs parameter limits document selection."""
        summaries = [
            (Document(
                page_content=f"Summary {i}",
                metadata={"source": f"doc{i}.txt", "type": "summary"}
            ), 0.1 * i)
            for i in range(5)
        ]
        
        mock_vector_store.similarity_search_with_score.return_value = summaries
        
        retriever = MetadataHierarchicalRetriever(
            vector_store=mock_vector_store,
            n_docs=2
        )
        
        # The first call should request only n_docs summaries
        retriever.get_relevant_documents("test query")
        
        first_call = mock_vector_store.similarity_search_with_score.call_args_list[0]
        assert first_call[1]["k"] == 2
        assert first_call[1]["filter"] == {"type": "summary"}
    
    def test_get_relevant_documents_respects_n_chunks_per_doc(self, mock_vector_store):
        """Test that n_chunks_per_doc parameter limits chunk selection."""
        summaries = [
            (Document(
                page_content="Summary",
                metadata={"source": "doc1.txt", "type": "summary"}
            ), 0.1)
        ]
        
        def side_effect(query, k, filter):
            if filter.get("type") == "summary":
                return summaries
            return []
        
        mock_vector_store.similarity_search_with_score.side_effect = side_effect
        
        retriever = MetadataHierarchicalRetriever(
            vector_store=mock_vector_store,
            n_docs=1,
            n_chunks_per_doc=7
        )
        
        retriever.get_relevant_documents("test query")
        
        # Check that chunks were requested with correct k
        chunk_calls = [
            call for call in mock_vector_store.similarity_search_with_score.call_args_list
            if call[1]["filter"].get("type") == "chunk"
        ]
        
        assert len(chunk_calls) == 1
        assert chunk_calls[0][1]["k"] == 7
    
    def test_get_relevant_documents_simple(self, mock_vector_store):
        """Test the simplified version that returns only documents."""
        summaries = [
            (Document(
                page_content="Summary",
                metadata={"source": "doc1.txt", "type": "summary"}
            ), 0.1)
        ]
        
        chunks = [
            (Document(
                page_content="Chunk 1",
                metadata={"source": "doc1.txt", "type": "chunk", "chunk_index": 0}
            ), 0.05)
        ]
        
        def side_effect(query, k, filter):
            if filter.get("type") == "summary":
                return summaries
            return chunks
        
        mock_vector_store.similarity_search_with_score.side_effect = side_effect
        
        retriever = MetadataHierarchicalRetriever(vector_store=mock_vector_store)
        results = retriever.get_relevant_documents_simple("test query")
        
        # Should return just documents, not tuples
        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content == "Chunk 1"
    
    def test_get_relevant_documents_no_source_in_summary(self, mock_vector_store):
        """Test handling of summaries without source metadata."""
        summaries = [
            (Document(
                page_content="Summary without source",
                metadata={"type": "summary"}  # Missing source
            ), 0.1)
        ]
        
        mock_vector_store.similarity_search_with_score.return_value = summaries
        
        retriever = MetadataHierarchicalRetriever(vector_store=mock_vector_store)
        results = retriever.get_relevant_documents("test query")
        
        # Should return empty results since no source to filter chunks
        assert results == []
        
        # Should only call once for summaries, not for chunks
        assert mock_vector_store.similarity_search_with_score.call_count == 1
