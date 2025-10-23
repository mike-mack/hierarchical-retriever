"""
Tests for the embeddings module.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from langchain_core.documents import Document
from app.embeddings import (
    validate_file,
    process_document,
    FileValidationError
)


class TestFileValidation:
    """Test file validation functionality."""
    
    def test_validate_text_file_success(self, sample_text_file):
        """Test successful validation of a text file."""
        result = validate_file(sample_text_file)
        
        assert result["valid"] is True
        assert result["file_name"] == "test.txt"
        assert result["file_extension"] == ".txt"
        assert result["file_size_bytes"] > 0
        assert "file_path" in result
    
    def test_validate_markdown_file_success(self, sample_markdown_file):
        """Test successful validation of a markdown file."""
        result = validate_file(sample_markdown_file)
        
        assert result["valid"] is True
        assert result["file_name"] == "test.md"
        assert result["file_extension"] == ".md"
        assert result["file_size_bytes"] > 0
    
    def test_validate_nonexistent_file(self):
        """Test validation fails for nonexistent file."""
        with pytest.raises(FileValidationError, match="File does not exist"):
            validate_file("/path/to/nonexistent/file.txt")
    
    def test_validate_empty_file(self, empty_file):
        """Test validation fails for empty file."""
        with pytest.raises(FileValidationError, match="File is empty"):
            validate_file(empty_file)
    
    def test_validate_file_too_large(self, large_text_file):
        """Test validation fails for file exceeding size limit."""
        with pytest.raises(FileValidationError, match="File too large"):
            validate_file(large_text_file, max_size_mb=50)
    
    def test_validate_unsupported_extension(self, temp_upload_dir):
        """Test validation fails for unsupported file extension."""
        file_path = Path(temp_upload_dir) / "test.exe"
        file_path.write_text("malicious content")
        
        with pytest.raises(FileValidationError, match="Unsupported file type"):
            validate_file(str(file_path))
    
    def test_validate_suspicious_path_characters(self, temp_upload_dir):
        """Test validation fails for suspicious path characters."""
        # The check happens before file existence, so we test with various suspicious patterns
        suspicious_paths = [
            str(Path(temp_upload_dir) / "file;command.txt"),
            str(Path(temp_upload_dir) / "file|pipe.txt"),
            str(Path(temp_upload_dir) / "file&background.txt"),
        ]
        
        for suspicious_path in suspicious_paths:
            # Create the file so we get past the existence check
            Path(suspicious_path).write_text("content")
            
            with pytest.raises(FileValidationError, match="Suspicious characters"):
                validate_file(suspicious_path)
    
    def test_validate_directory_not_file(self, temp_upload_dir):
        """Test validation fails when path is a directory."""
        with pytest.raises(FileValidationError, match="Path is not a file"):
            validate_file(temp_upload_dir)


class TestProcessDocument:
    """Test document processing functionality."""
    
    @patch('app.embeddings.get_vector_store')
    @patch('app.embeddings.TextLoader')
    def test_process_text_document(self, mock_loader, mock_get_vector_store, sample_text_file, sample_documents):
        """Test processing a text document."""
        # Setup mocks - now only one vector store call
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_loader_instance
        
        # Process document
        result = process_document(sample_text_file)
        
        # Assertions - new return format
        assert result["success"] is True
        assert result["source_id"] == "test.txt"
        assert result["summary_stored"] is True
        assert result["chunks_stored"] > 0
        assert result["collection"] == "hierarchical_documents"
        
        # Verify vector store was called twice: once for summary, once for chunks
        assert mock_vector_store.add_documents.call_count == 2
    
    @patch('app.embeddings.get_vector_store')
    @patch('app.embeddings.UnstructuredMarkdownLoader')
    def test_process_markdown_document(self, mock_loader, mock_get_vector_store, sample_markdown_file, sample_documents):
        """Test processing a markdown document."""
        # Setup mocks - now only one vector store call
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_loader_instance
        
        # Process document
        result = process_document(sample_markdown_file)
        
        # Assertions - new return format
        assert result["success"] is True
        assert result["source_id"] == "test.md"
        assert result["summary_stored"] is True
        assert result["chunks_stored"] > 0
        assert mock_vector_store.add_documents.call_count == 2
    
    @patch('app.embeddings.get_vector_store')
    def test_process_invalid_document(self, mock_get_vector_store, empty_file):
        """Test processing an invalid document raises error."""
        with pytest.raises(FileValidationError):
            process_document(empty_file)
    
    @patch('app.embeddings.get_vector_store')
    def test_process_unsupported_file_type(self, mock_get_vector_store, temp_upload_dir):
        """Test processing unsupported file type raises error."""
        file_path = Path(temp_upload_dir) / "test.docx"
        file_path.write_text("content")
        
        with pytest.raises(FileValidationError):
            process_document(str(file_path))
    
    @patch('app.embeddings.get_vector_store')
    @patch('app.embeddings.TextLoader')
    def test_document_chunking(self, mock_loader, mock_get_vector_store, sample_text_file, sample_documents):
        """Test that documents are properly chunked."""
        # Setup mocks - now only one vector store call
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        
        # Create a longer document for chunking
        long_doc = [Document(
            page_content="A" * 1000,  # Long enough to be split into chunks
            metadata={"source": "long.txt"}
        )]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = long_doc
        mock_loader.return_value = mock_loader_instance
        
        # Process document
        result = process_document(sample_text_file)
        
        # Verify chunks were created in new format
        assert result["chunks_stored"] > 0
        
        # Verify vector store received documents (summary + chunks)
        assert mock_vector_store.add_documents.call_count == 2
        
        # Get the second call (chunks) and verify
        chunks_call = mock_vector_store.add_documents.call_args_list[1]
        chunks_added = chunks_call[0][0]
        assert len(chunks_added) > 0
        
        # Verify metadata includes type="chunk"
        for chunk in chunks_added:
            assert chunk.metadata.get("type") == "chunk"
            assert "chunk_index" in chunk.metadata
