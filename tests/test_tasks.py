"""
Tests for the tasks module.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.tasks import process_file_task, celery_app


class TestCeleryConfiguration:
    """Test Celery app configuration."""
    
    def test_celery_app_exists(self):
        """Test that Celery app is properly initialized."""
        assert celery_app is not None
        assert celery_app.main == "tasks"
    
    def test_celery_broker_configured(self):
        """Test that Celery broker is configured."""
        assert "redis://redis:6379/0" in celery_app.conf.broker_url
    
    def test_celery_backend_configured(self):
        """Test that Celery result backend is configured."""
        assert "redis://redis:6379/0" in celery_app.conf.result_backend


class TestProcessFileTask:
    """Test the process_file_task."""
    
    @patch('app.tasks.process_document')
    def test_process_file_task_success(self, mock_process_document):
        """Test successful file processing task."""
        mock_result = {
            "num_chunks": 5,
            "file_type": ".txt",
            "message": "Documents processed and stored successfully"
        }
        mock_process_document.return_value = mock_result
        
        # Call the task function directly (not as Celery task)
        result = process_file_task.apply(args=["/path/to/test.txt"]).result
        
        assert result == mock_result
        mock_process_document.assert_called_once_with("/path/to/test.txt")
    
    @patch('app.tasks.process_document')
    def test_process_file_task_with_error(self, mock_process_document):
        """Test file processing task with error."""
        mock_process_document.side_effect = Exception("Processing error")
        
        # The task should catch and handle the exception
        result = process_file_task.apply(args=["/path/to/test.txt"])
        
        # Check that the task failed
        assert result.state == "FAILURE"
        assert "Processing error" in str(result.info)
    
    @patch('app.tasks.process_document')
    def test_process_file_task_registered(self, mock_process_document):
        """Test that the task is properly registered with Celery."""
        assert "app.tasks.process_file_task" in celery_app.tasks
    
    @patch('app.tasks.process_document')
    def test_process_file_task_returns_embeddings_result(self, mock_process_document):
        """Test that task returns the embeddings processing result."""
        expected_result = {
            "num_chunks": 10,
            "file_type": ".md",
            "file_info": {"file_name": "test.md"},
            "message": "Documents processed and stored successfully"
        }
        mock_process_document.return_value = expected_result
        
        result = process_file_task.apply(args=["/path/to/test.md"]).result
        
        assert result == expected_result
        assert result["num_chunks"] == 10
        assert result["file_type"] == ".md"
