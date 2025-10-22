from celery import Celery
from app.embeddings import process_document


celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)


@celery_app.task
def process_file_task(file_path: str):
    embeddings_result = process_document(file_path)
    return embeddings_result
