# Hierarchical Retriever

A prototype application demonstrating hierarchical retrieval for Retrieval-Augmented Generation (RAG). This system uses a two-stage retrieval approach to efficiently search through large document collections by first identifying relevant documents, then finding the most relevant chunks within those documents.

## Overview

The hierarchical retrieval approach improves search accuracy and efficiency by:
1. **Coarse Retrieval**: First searching at the document level to identify the most relevant documents
2. **Fine Retrieval**: Then searching within those documents at the chunk level for precise information

This two-stage approach reduces computational overhead while maintaining high retrieval quality.

## Structure

The application consists of several components:

- **FastAPI Web Server**: Provides HTTP endpoints for file upload, document management, and querying
- **Celery Worker**: Handles asynchronous document processing tasks
- **PostgreSQL with pgvector**: Stores embeddings with vector similarity search capabilities
- **Redis**: Message broker for Celery task queue
- **Ollama**: Local LLM embeddings using `nomic-embed-text` model

### Key Features

- **Multi-format Support**: Process `.txt`, `.md`, and `.pdf` files
- **File Validation**: Security checks including MIME type verification and size limits
- **Asynchronous Processing**: Non-blocking file processing with Celery
- **Two-level Vector Storage**: Separate collections for document-level and chunk-level embeddings
- **Interactive UI**: HTMX-powered interface for uploads and queries
- **Hierarchical Search**: Efficient two-stage retrieval system

## Getting Started

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.ai) installed locally with `nomic-embed-text` model

```bash
# Pull the embedding model
ollama pull nomic-embed-text
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mike-mack/hierarchical-retriever.git
cd hierarchical-retriever
```

2. Start the services:
```bash
docker-compose up --build
```

3. Access the application at `http://localhost:8000`

## 📖 Usage

### Upload Documents

1. Navigate to `http://localhost:8000`
2. Click "Choose File" and select a `.txt`, `.md`, or `.pdf` file
3. Click "Upload" to process the file asynchronously
4. Check the task status to monitor progress

### Query Documents

1. Navigate to `http://localhost:8000/query`
2. Enter your search query
3. Optionally adjust the number of results (1-20)
4. View ranked results with similarity scores

### View Processed Documents

Navigate to `http://localhost:8000/documents` or click the link in the query interface to see all uploaded documents.

## Project Structure

```
hierarchical-retriever/
├── app/
│   ├── main.py          # FastAPI application and endpoints
│   ├── retrievers.py    # HierarchicalRetriever implementation
│   ├── vectorstore.py   # PGVector configuration
│   ├── embeddings.py    # Document processing and validation
│   └── tasks.py         # Celery task definitions
├── uploads/             # Uploaded files directory
├── docker-compose.yml   # Multi-container Docker configuration
├── Dockerfile          # Application container definition
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technical Details

### Hierarchical Retrieval Process

The `HierarchicalRetriever` class implements the two-stage approach:

```python
class HierarchicalRetriever:
    def get_relevant_documents(self, query: str):
        # Stage 1: Coarse retrieval - find relevant documents
        coarse_docs = self.doc_store.similarity_search(query, k=self.n_docs)
        
        # Stage 2: Fine retrieval - find relevant chunks within those docs
        relevant_chunks = []
        for doc in coarse_docs:
            doc_id = doc.metadata.get("source")
            fine_chunks = self.chunk_store.similarity_search_with_score(
                query, k=self.n_chunks, filter={"source": doc_id}
            )
            relevant_chunks.extend(fine_chunks)
        
        return relevant_chunks
```

### Document Processing Pipeline

1. **Validation**: File size, type, MIME type, and security checks
2. **Loading**: Format-specific document loaders (TextLoader, UnstructuredMarkdownLoader, PyPDFLoader)
3. **Storage**: Full document stored in `doc_level_embeddings` collection
4. **Chunking**: Documents split into 500-character chunks with 50-character overlap
5. **Indexing**: Chunks stored in `chunk_level_embeddings` collection with source metadata

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql://user:password@postgres:5432/embeddings_db`)
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: `http://host.docker.internal:11434`)

## 🛠️ Development

### Running Tests

```bash
# Add test command when tests are implemented
pytest
```

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis and PostgreSQL separately
# Update DATABASE_URL environment variable

# Run FastAPI server
uvicorn app.main:app --reload

# Run Celery worker in separate terminal
celery -A app.tasks worker --loglevel=info
```

## Dependencies

Key libraries used:
- **FastAPI**: Modern web framework for building APIs
- **LangChain**: Framework for building LLM applications
- **pgvector**: PostgreSQL extension for vector similarity search
- **Celery**: Distributed task queue
- **Ollama**: Local LLM embeddings
- **HTMX**: Lightweight JavaScript framework for dynamic UIs

See `requirements.txt` for full dependency list.
