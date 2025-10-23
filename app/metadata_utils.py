"""
Utility functions for working with hierarchical retrieval metadata.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


def get_document_summaries(vector_store, source_ids: Optional[List[str]] = None) -> List[Document]:
    """
    Retrieve document summaries from the vector store.
    
    Args:
        vector_store: The PGVector store
        source_ids: Optional list of specific source IDs to retrieve
        
    Returns:
        List of Document objects with type="summary"
    """
    if source_ids:
        summaries = []
        for source_id in source_ids:
            results = vector_store.similarity_search(
                "",  # Empty query since we're filtering by metadata
                k=1,
                filter={"type": "summary", "source": source_id}
            )
            summaries.extend(results)
        return summaries
    else:
        # Get all summaries (using a generic search)
        return vector_store.similarity_search(
            "",
            k=100,  # Adjust as needed
            filter={"type": "summary"}
        )


def get_document_chunks(vector_store, source_id: str) -> List[Document]:
    """
    Retrieve all chunks for a specific document.
    
    Args:
        vector_store: The PGVector store
        source_id: The source identifier
        
    Returns:
        List of Document objects with type="chunk" for the given source
    """
    chunks = vector_store.similarity_search(
        "",  # Empty query since we're filtering by metadata
        k=1000,  # Large number to get all chunks
        filter={"type": "chunk", "source": source_id}
    )
    
    # Sort by chunk_index if available
    chunks.sort(key=lambda doc: doc.metadata.get("chunk_index", 0))
    
    return chunks


def reconstruct_document(vector_store, source_id: str) -> Dict[str, Any]:
    """
    Reconstruct a full document by retrieving its summary and all chunks.
    
    Args:
        vector_store: The PGVector store
        source_id: The source identifier
        
    Returns:
        Dict containing summary and chunks
    """
    summary_docs = vector_store.similarity_search(
        "",
        k=1,
        filter={"type": "summary", "source": source_id}
    )
    
    chunks = get_document_chunks(vector_store, source_id)
    
    return {
        "source_id": source_id,
        "summary": summary_docs[0] if summary_docs else None,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }


def list_all_documents(vector_store) -> List[str]:
    """
    Get a list of all unique document source IDs in the vector store.
    
    Args:
        vector_store: The PGVector store
        
    Returns:
        List of unique source IDs
    """
    summaries = get_document_summaries(vector_store)
    source_ids: set[str] = set()
    for doc in summaries:
        source = doc.metadata.get("source")
        if source and isinstance(source, str):
            source_ids.add(source)
    return sorted(list(source_ids))


def get_metadata_stats(vector_store) -> Dict[str, Any]:
    """
    Get statistics about the metadata in the vector store.
    
    Args:
        vector_store: The PGVector store
        
    Returns:
        Dict with statistics
    """
    # Get summaries
    summaries = vector_store.similarity_search(
        "",
        k=1000,
        filter={"type": "summary"}
    )
    
    # Get chunks
    chunks = vector_store.similarity_search(
        "",
        k=10000,
        filter={"type": "chunk"}
    )
    
    unique_sources = set()
    for doc in summaries + chunks:
        if doc.metadata.get("source"):
            unique_sources.add(doc.metadata.get("source"))
    
    return {
        "total_summaries": len(summaries),
        "total_chunks": len(chunks),
        "unique_documents": len(unique_sources),
        "avg_chunks_per_doc": len(chunks) / len(unique_sources) if unique_sources else 0,
        "source_ids": sorted(list(unique_sources))
    }


# Example usage
if __name__ == "__main__":
    from app.vectorstore import get_vector_store
    
    vector_store = get_vector_store("hierarchical_documents")
    
    # Get statistics
    stats = get_metadata_stats(vector_store)
    print("Vector Store Statistics:")
    print(f"  Total Summaries: {stats['total_summaries']}")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Unique Documents: {stats['unique_documents']}")
    print(f"  Avg Chunks/Doc: {stats['avg_chunks_per_doc']:.2f}")
    print(f"  Source IDs: {stats['source_ids']}")
    
    # List all documents
    print("\n\nAll Documents:")
    docs = list_all_documents(vector_store)
    for doc_id in docs:
        print(f"  - {doc_id}")
    
    # Reconstruct a specific document
    if docs:
        print(f"\n\nReconstructing document: {docs[0]}")
        reconstructed = reconstruct_document(vector_store, docs[0])
        print(f"  Summary: {reconstructed['summary'].page_content[:100] if reconstructed['summary'] else 'None'}...")
        print(f"  Total Chunks: {reconstructed['total_chunks']}")
