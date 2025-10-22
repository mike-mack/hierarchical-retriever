# Test Suite for Hierarchical Retriever

This directory contains comprehensive tests for the hierarchical retriever application.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_embeddings.py          # Tests for embeddings and file processing
├── test_main.py                # Tests for FastAPI endpoints
├── test_retrievers.py          # Tests for hierarchical retrieval
├── test_tasks.py               # Tests for Celery tasks
└── test_vectorstore.py         # Tests for vector store creation
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_embeddings.py -v
```

### Run specific test class
```bash
pytest tests/test_embeddings.py::TestFileValidation -v
```

### Run specific test
```bash
pytest tests/test_embeddings.py::TestFileValidation::test_validate_text_file_success -v
```

### Run with coverage
```bash
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
```

## Test Coverage

The test suite covers:

### 1. File Validation (`test_embeddings.py`)
- ✅ Valid text files (.txt)
- ✅ Valid markdown files (.md)
- ✅ Nonexistent files
- ✅ Empty files
- ✅ Files exceeding size limits
- ✅ Unsupported file extensions
- ✅ Suspicious path characters (security)
- ✅ Directory vs. file validation

### 2. Document Processing (`test_embeddings.py`)
- ✅ Text document processing
- ✅ Markdown document processing
- ✅ Invalid document handling
- ✅ Unsupported file type handling
- ✅ Document chunking

### 3. FastAPI Endpoints (`test_main.py`)
- ✅ Index page rendering
- ✅ Upload form presence
- ✅ File upload success
- ✅ Upload directory creation
- ✅ Task status checking (pending, success, failure)
- ✅ Document listing
- ✅ Error handling (missing DB, DB errors)

### 4. Hierarchical Retriever (`test_retrievers.py`)
- ✅ Initialization with default parameters
- ✅ Initialization with custom parameters
- ✅ Empty search results
- ✅ Search with results
- ✅ Respect for n_docs parameter
- ✅ Respect for n_chunks parameter
- ✅ Handling missing source metadata

### 5. Celery Tasks (`test_tasks.py`)
- ✅ Celery app configuration
- ✅ Broker configuration
- ✅ Backend configuration
- ✅ Task execution success
- ✅ Task execution with errors
- ✅ Task registration
- ✅ Task result handling

### 6. Vector Store (`test_vectorstore.py`)
- ✅ Default collection creation
- ✅ Custom collection creation
- ✅ Environment variable usage
- ✅ Default Ollama URL fallback

## Fixtures

### File Fixtures
- `temp_upload_dir`: Temporary directory for file uploads
- `sample_text_file`: Sample .txt file
- `sample_markdown_file`: Sample .md file
- `large_text_file`: File exceeding size limits
- `empty_file`: Empty file for validation tests

### Mock Fixtures
- `mock_vector_store`: Mocked vector store
- `mock_embeddings`: Mocked embeddings
- `sample_documents`: Sample LangChain documents
- `sample_chunks`: Sample document chunks
- `mock_celery_task`: Mocked Celery task

### Environment Fixtures
- `mock_env_vars`: Auto-used fixture to set test environment variables

## Test Statistics

- **Total Tests**: 41
- **Pass Rate**: 100%
- **Test Files**: 5
- **Test Classes**: 14

## Notes

- Tests use mocking to avoid external dependencies (database, Redis, Ollama)
- The `libmagic` import is made optional to support testing without system libraries
- Environment variables are automatically mocked for all tests
- Tests are isolated and can run in any order
