import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from app.retrievers import MetadataHierarchicalRetriever
from app.tasks import process_file_task
import os, shutil
from pathlib import Path
from sqlalchemy import create_engine, text
from .vectorstore import get_vector_store

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>File Upload</title>
        <script src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.7/dist/htmx.min.js"></script>
    </head>
    <body style="font-family:sans-serif; padding:2rem;">
        <h2>Upload a file</h2>
        <form
            hx-post="/upload"
            hx-target="#result"
            hx-swap="innerHTML"
            enctype="multipart/form-data"
        >
            <input type="file" name="file" required>
            <button type="submit">Upload</button>
        </form>

        <div id="result" style="margin-top:1rem;"></div>
    </body>
    </html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...)):
    # Save file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Kick off async task
    task = process_file_task.delay(file_path)

    # Return simple HTML snippet
    return f"""
    <p>✅ File uploaded: <b>{file.filename}</b></p>
    <p>Task started with ID: <code>{task.id}</code></p>
    <p><a hx-get="/task-status/{task.id}" hx-target="#result" hx-swap="innerHTML">
        Check status
    </a></p>
    """

@app.get("/task-status/{task_id}", response_class=HTMLResponse)
async def task_status(task_id: str):
    from app.tasks import celery_app
    result = celery_app.AsyncResult(task_id)
    return f"""
    <p>Task ID: <code>{task_id}</code></p>
    <p>Status: <b>{result.status}</b></p>
    <p>Result: {result.result}</p>
    <a hx-get="/task-status/{task_id}" hx-target="#result" hx-swap="innerHTML">🔄 Refresh</a>
    """

@app.get("/documents", response_class=HTMLResponse)
def list_documents():
    """
    Returns a list of unique documents that have been processed and stored.
    Each document includes its source path and filename.
    """
    try:
        # vector_store = get_vector_store("doc_level_embeddings")
        
        # Get the database connection to query metadata directly
        connection_string = os.getenv("DATABASE_URL")
        
        if not connection_string:
            return """
            <div style="color: red;">
                <p>❌ Error: DATABASE_URL not configured</p>
            </div>
            """
        
        engine = create_engine(connection_string)
        
        # Query for unique document sources from the langchain_pg_embedding table
        # The metadata is stored as JSONB with a 'source' field
        with engine.connect() as conn:
            query = text("""
                SELECT DISTINCT cmetadata->>'source' as source
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = 'hierarchical_documents'
                )
                AND cmetadata->>'source' IS NOT NULL
                ORDER BY source
            """)
            result = conn.execute(query)
            sources = [row[0] for row in result]
        
        # Build HTML list
        if not sources:
            return """
            <div>
                <h3>📄 Processed Documents</h3>
                <p>No documents have been processed yet.</p>
            </div>
            """
        
        # Create HTML list items
        html_items = []
        for source in sources:
            path = Path(source)
            extension = path.suffix.lower()
            
            # Add emoji based on file type
            emoji = "📄"
            if extension == ".pdf":
                emoji = "📕"
            elif extension == ".md":
                emoji = "📝"
            elif extension == ".txt":
                emoji = "📃"
            
            html_items.append(f"""
                <li>
                    <span style="font-size: 1.2em;">{emoji}</span>
                    <strong>{path.name}</strong>
                    <span style="color: #666; font-size: 0.9em;">({extension})</span>
                </li>
            """)
        
        html = f"""
        <div>
            <h3>📄 Processed Documents ({len(sources)})</h3>
            <ul style="list-style: none; padding-left: 0;">
                {''.join(html_items)}
            </ul>
        </div>
        """
        
        return html
    
    except Exception as e:
        return f"""
        <div style="color: red;">
            <p>❌ Error: {str(e)}</p>
        </div>
        """

@app.get("/query", response_class=HTMLResponse)
def query_form():
    """
    Display a form to query the vector store.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Query Documents</title>
        <script src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.7/dist/htmx.min.js"></script>
    </head>
    <body style="font-family:sans-serif; padding:2rem;">
        <h2>🔍 Query Processed Documents</h2>
        <form
            hx-post="/query"
            hx-target="#query-results"
            hx-swap="innerHTML"
        >
            <div style="margin-bottom: 1rem;">
                <label for="query">Enter your query:</label><br>
                <input 
                    type="text" 
                    id="query" 
                    name="query" 
                    required 
                    style="width: 400px; padding: 0.5rem; margin-top: 0.5rem;"
                    placeholder="e.g., What are the factions?"
                >
            </div>
            <div style="margin-bottom: 1rem;">
                <label for="k">Number of results:</label><br>
                <input 
                    type="number" 
                    id="k" 
                    name="k" 
                    value="5" 
                    min="1" 
                    max="20"
                    style="width: 100px; padding: 0.5rem; margin-top: 0.5rem;"
                >
            </div>
            <button type="submit" style="padding: 0.5rem 1rem; cursor: pointer;">Search</button>
        </form>

        <div id="query-results" style="margin-top: 2rem;"></div>
        
        <div style="margin-top: 2rem;">
            <a href="/">← Back to Upload</a> | 
            <a hx-get="/documents" hx-target="#query-results" hx-swap="innerHTML">View All Documents</a>
        </div>
    </body>
    </html>
    """

@app.post("/query", response_class=HTMLResponse)
def query_documents(query: str = Form(...), k: int = Form(5)):
    """
    Query the vector store for similar documents.
    
    Args:
        query: The search query string
        k: Number of results to return (default: 5)
    """
    try:
        if not query or query.strip() == "":
            return """
            <div style="color: orange;">
                <p>⚠️ Please enter a query.</p>
            </div>
            """
        
        # Limit k to reasonable bounds
        k = max(1, min(k, 20))
        
        # Use the new metadata-based hierarchical retriever
        vector_store = get_vector_store("hierarchical_documents")
        
        # Calculate n_docs and n_chunks_per_doc from k
        # For k results, retrieve from ~k/3 documents with 3-5 chunks each
        n_docs = max(2, k // 3)
        n_chunks_per_doc = max(3, k // n_docs)
        
        retriever = MetadataHierarchicalRetriever(
            vector_store=vector_store,
            n_docs=n_docs,
            n_chunks_per_doc=n_chunks_per_doc
        )
        results = retriever.get_relevant_documents(query)
        
        if not results:
            return f"""
            <div>
                <h3>🔍 Search Results for: "{query}"</h3>
                <p>No results found. Try a different query or upload more documents.</p>
            </div>
            """
        
        # Build HTML results
        html_results = []
        for i, (doc, score) in enumerate(results, 1):
            # Extract metadata
            source = doc.metadata.get('source', 'Unknown')
            source_path = Path(source)
            chunk_type = doc.metadata.get('type', 'unknown')
            chunk_index = doc.metadata.get('chunk_index', 'N/A')
            parent_score = doc.metadata.get('parent_summary_score', None)
            
            # Calculate similarity percentage (lower score = more similar)
            # Note: The score is a distance metric, so lower is better
            similarity_pct = max(0, 100 - (score * 10))
            
            # Build metadata badge
            metadata_info = f"Type: {chunk_type}"
            if chunk_index != 'N/A':
                metadata_info += f" | Chunk #{chunk_index}"
            if parent_score is not None:
                parent_sim_pct = max(0, 100 - (parent_score * 10))
                metadata_info += f" | Doc Relevance: {parent_sim_pct:.1f}%"
            
            html_results.append(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; margin-bottom: 1rem; border-radius: 5px;">
                <div style="margin-bottom: 0.5rem;">
                    <strong>Result #{i}</strong> - 
                    <span style="color: #0066cc;">{source_path.name}</span>
                    <span style="color: #666; font-size: 0.9em;">(Chunk Similarity: {similarity_pct:.1f}%)</span>
                </div>
                <div style="font-size: 0.85em; color: #888; margin-bottom: 0.5rem;">
                    {metadata_info}
                </div>
                <div style="background-color: #f5f5f5; padding: 0.75rem; border-radius: 3px; white-space: pre-wrap; font-family: monospace; font-size: 0.9em;">
{doc.page_content}
                </div>
            </div>
            """)
        
        html = f"""
        <div>
            <h3>🔍 Search Results for: "{query}"</h3>
            <p>Found {len(results)} result(s)</p>
            {''.join(html_results)}
        </div>
        """
        
        return html
    
    except Exception as e:

        traceback.print_exc()
        print(e)
        return f"""
        <div style="color: red;">
            <p>❌ Error: {str(e)}</p>
        </div>
        """
