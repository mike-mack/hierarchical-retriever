"""
Pytest configuration and fixtures for the hierarchical retriever tests.
"""
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock
from langchain_core.documents import Document


@pytest.fixture
def temp_upload_dir():
    """Create a temporary upload directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_text_file(temp_upload_dir):
    """Create a sample text file for testing."""
    file_path = Path(temp_upload_dir) / "test.txt"
    file_path.write_text("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
    return str(file_path)


@pytest.fixture
def sample_markdown_file(temp_upload_dir):
    """Create a sample markdown file for testing."""
    file_path = Path(temp_upload_dir) / "test.md"
    content = """# Test Document

This is a test markdown document.

## Section 1
Content for section 1.

## Section 2
Content for section 2.
"""
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def large_text_file(temp_upload_dir):
    """Create a large text file for testing size limits."""
    file_path = Path(temp_upload_dir) / "large.txt"
    # Create a file larger than 50MB
    with open(file_path, "w") as f:
        # Write about 51MB of data
        for _ in range(51 * 1024):
            f.write("A" * 1024 + "\n")
    return str(file_path)


@pytest.fixture
def empty_file(temp_upload_dir):
    """Create an empty file for testing."""
    file_path = Path(temp_upload_dir) / "empty.txt"
    file_path.touch()
    return str(file_path)


@pytest.fixture
def suspicious_file(temp_upload_dir):
    """Create a file with suspicious path characters."""
    file_path = Path(temp_upload_dir) / "../suspicious.txt"
    return str(file_path)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.add_documents = MagicMock(return_value=None)
    store.similarity_search = MagicMock(return_value=[])
    store.similarity_search_with_score = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    embeddings = MagicMock()
    embeddings.embed_documents = MagicMock(return_value=[[0.1] * 768])
    embeddings.embed_query = MagicMock(return_value=[0.1] * 768)
    return embeddings


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is the first document about machine learning.",
            metadata={"source": "doc1.txt", "page": 0}
        ),
        Document(
            page_content="This is the second document about artificial intelligence.",
            metadata={"source": "doc2.txt", "page": 0}
        ),
        Document(
            page_content="This is the third document about neural networks.",
            metadata={"source": "doc3.txt", "page": 0}
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "doc1.txt", "chunk_id": 0}
        ),
        Document(
            page_content="It involves training models on data.",
            metadata={"source": "doc1.txt", "chunk_id": 1}
        ),
        Document(
            page_content="AI can solve complex problems.",
            metadata={"source": "doc2.txt", "chunk_id": 0}
        ),
    ]


@pytest.fixture
def mock_celery_task():
    """Create a mock Celery task."""
    task = MagicMock()
    task.id = "test-task-id-12345"
    task.status = "PENDING"
    task.result = None
    return task


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up environment variables for tests."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
