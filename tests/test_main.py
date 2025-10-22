"""
Tests for the main FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
import io
from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestIndexEndpoint:
    """Test the index (/) endpoint."""
    
    def test_index_returns_html(self, client):
        """Test that index endpoint returns HTML."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Upload a file" in response.text
        assert "htmx" in response.text
    
    def test_index_contains_upload_form(self, client):
        """Test that index contains file upload form."""
        response = client.get("/")
        
        assert '<form' in response.text
        assert 'hx-post="/upload"' in response.text
        assert 'type="file"' in response.text


class TestUploadEndpoint:
    """Test the upload (/upload) endpoint."""
    
    @patch('app.main.process_file_task')
    @patch('app.main.shutil.copyfileobj')
    def test_upload_file_success(self, mock_copyfileobj, mock_task, client, temp_upload_dir, monkeypatch):
        """Test successful file upload."""
        # Set upload directory to temp directory
        monkeypatch.setattr("app.main.UPLOAD_DIR", temp_upload_dir)
        
        # Mock the Celery task
        mock_task_result = MagicMock()
        mock_task_result.id = "test-task-123"
        mock_task.delay.return_value = mock_task_result
        
        # Create a test file
        file_content = b"Test file content"
        file = ("test.txt", io.BytesIO(file_content), "text/plain")
        
        response = client.post(
            "/upload",
            files={"file": file}
        )
        
        assert response.status_code == 200
        assert "test.txt" in response.text
        assert "test-task-123" in response.text
        assert "Task started with ID" in response.text
        
        # Verify task was called
        mock_task.delay.assert_called_once()
    
    @patch('app.main.process_file_task')
    def test_upload_creates_upload_dir(self, mock_task, client):
        """Test that upload directory exists and upload works."""
        import os
        
        mock_task_result = MagicMock()
        mock_task_result.id = "test-task-456"
        mock_task.delay.return_value = mock_task_result
        
        file_content = b"Test content"
        file = ("test.txt", io.BytesIO(file_content), "text/plain")
        
        response = client.post("/upload", files={"file": file})
        
        assert response.status_code == 200
        # Verify the default upload directory exists (created at module load)
        assert os.path.exists("uploads")


class TestTaskStatusEndpoint:
    """Test the task-status endpoint."""
    
    @patch('app.tasks.celery_app')
    def test_task_status_pending(self, mock_celery_app, client):
        """Test task status endpoint with pending task."""
        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_result.result = None
        mock_celery_app.AsyncResult.return_value = mock_result
        
        response = client.get("/task-status/test-task-id")
        
        assert response.status_code == 200
        assert "test-task-id" in response.text
        assert "PENDING" in response.text
        assert "Refresh" in response.text
    
    @patch('app.tasks.celery_app')
    def test_task_status_success(self, mock_celery_app, client):
        """Test task status endpoint with successful task."""
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.result = {"num_chunks": 5, "message": "Success"}
        mock_celery_app.AsyncResult.return_value = mock_result
        
        response = client.get("/task-status/test-task-id")
        
        assert response.status_code == 200
        assert "SUCCESS" in response.text
        assert "num_chunks" in response.text or "5" in response.text
    
    @patch('app.tasks.celery_app')
    def test_task_status_failure(self, mock_celery_app, client):
        """Test task status endpoint with failed task."""
        mock_result = MagicMock()
        mock_result.status = "FAILURE"
        mock_result.result = Exception("Processing failed")
        mock_celery_app.AsyncResult.return_value = mock_result
        
        response = client.get("/task-status/test-task-id")
        
        assert response.status_code == 200
        assert "FAILURE" in response.text


class TestDocumentsEndpoint:
    """Test the documents listing endpoint."""
    
    @patch('app.main.create_engine')
    def test_list_documents_success(self, mock_create_engine, client):
        """Test successful document listing."""
        # Mock database connection and query results
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("uploads/doc1.txt",),
            ("uploads/doc2.md",),
        ]
        mock_conn.__enter__.return_value.execute.return_value = mock_result
        
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        response = client.get("/documents")
        
        assert response.status_code == 200
        # Should contain HTML response
        assert "text/html" in response.headers["content-type"]
    
    @patch('app.main.create_engine')
    def test_list_documents_no_database_url(self, mock_create_engine, client, monkeypatch):
        """Test document listing when DATABASE_URL is not set."""
        # Remove DATABASE_URL
        monkeypatch.delenv("DATABASE_URL", raising=False)
        
        response = client.get("/documents")
        
        assert response.status_code == 200
        assert "DATABASE_URL not configured" in response.text
    
    @patch('app.main.create_engine')
    def test_list_documents_database_error(self, mock_create_engine, client):
        """Test document listing when database query fails."""
        mock_create_engine.side_effect = Exception("Database connection failed")
        
        response = client.get("/documents")
        
        assert response.status_code == 200
        # Should contain error message
        assert "Error" in response.text or "error" in response.text.lower()
