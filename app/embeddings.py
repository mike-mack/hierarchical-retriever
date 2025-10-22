import os
import magic
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader
)
from .vectorstore import get_vector_store


class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass


def validate_file(file_path: str, max_size_mb: int = 50) -> dict:
    """
    Validate a file before processing to ensure it's safe and valid.
    
    Args:
        file_path: Path to the file to validate
        max_size_mb: Maximum allowed file size in megabytes (default: 50MB)
    
    Returns:
        dict: Validation results including file info
    
    Raises:
        FileValidationError: If validation fails
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise FileValidationError(f"File does not exist: {file_path}")
    
    # Check if it's a file (not a directory)
    if not path.is_file():
        raise FileValidationError(f"Path is not a file: {file_path}")
    
    # Check file size
    file_size = path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size == 0:
        raise FileValidationError("File is empty")
    
    if file_size > max_size_bytes:
        raise FileValidationError(
            f"File too large: {file_size / 1024 / 1024:.2f}MB (max: {max_size_mb}MB)"
        )
    
    # Check file extension
    file_extension = path.suffix.lower()
    allowed_extensions = {'.txt', '.md', '.pdf'}
    
    if file_extension not in allowed_extensions:
        raise FileValidationError(
            f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Verify MIME type matches extension (security check)
    mime_type = "unknown"
    try:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)
        
        # Expected MIME types for each extension
        expected_mimes = {
            '.txt': ['text/plain', 'text/x-plain'],
            '.md': ['text/plain', 'text/markdown', 'text/x-markdown'],
            '.pdf': ['application/pdf']
        }
        
        if file_extension in expected_mimes:
            if mime_type not in expected_mimes[file_extension]:
                raise FileValidationError(
                    f"MIME type mismatch: file has extension {file_extension} but MIME type is {mime_type}"
                )
    except Exception as e:
        # If python-magic is not available, log warning but continue
        print(f"Warning: Could not verify MIME type: {e}")
    
    # Check for suspicious file names (basic security check)
    suspicious_patterns = ['..', '~', '$', '`', '|', ';', '&', '\x00']
    if any(pattern in str(path) for pattern in suspicious_patterns):
        raise FileValidationError("Suspicious characters detected in file path")
    
    # Check file permissions (ensure it's readable)
    if not os.access(file_path, os.R_OK):
        raise FileValidationError("File is not readable")
    
    return {
        "valid": True,
        "file_path": str(path.absolute()),
        "file_name": path.name,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / 1024 / 1024, 2),
        "file_extension": file_extension,
        "mime_type": mime_type
    }


def process_document(file_path: str):
    # Validate file first
    validation_result = validate_file(file_path)

    if not validation_result["valid"]:
        return {"error": validation_result["error"]}

    doc_store = get_vector_store("doc_level_embeddings")
    chunk_store = get_vector_store("chunk_level_embeddings")

    # Detect file type and load document
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.txt':
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_extension == '.md':
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .txt, .md, .pdf")
    
    # Load documents
    documents = loader.load()
    doc_store.add_documents(documents)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    split_documents = splitter.split_documents(documents)
    
    # Add documents to vector store
    chunk_store.add_documents(split_documents)
    
    return {
        "num_chunks": len(split_documents),
        "file_type": file_extension,
        "file_info": validation_result,
        "message": "Documents processed and stored successfully"
    }
