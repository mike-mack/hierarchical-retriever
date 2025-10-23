class HierarchicalRetriever:
    """
    Legacy implementation using separate vector stores for documents and chunks.
    """
    def __init__(self, doc_store, chunk_store, n_docs=3, n_chunks=5):
        self.doc_store = doc_store
        self.chunk_store = chunk_store
        self.n_docs = n_docs
        self.n_chunks = n_chunks

    def get_relevant_documents(self, query: str):
        # step 1: coarse retrieval
        coarse_docs = self.doc_store.similarity_search(query, k=self.n_docs)
        relevant_chunks = []
        # step 2: fine retrieval within each doc
        for doc in coarse_docs:
            doc_id = doc.metadata.get("source")
            fine_chunks = self.chunk_store.similarity_search_with_score(
                query, k=self.n_chunks, filter={"source": doc_id}
            )
            relevant_chunks.extend(fine_chunks)
        return relevant_chunks


class MetadataHierarchicalRetriever:
    """
    Hierarchical retrieval using a single vector store with metadata differentiation.
    
    This approach stores both summaries and chunks in the same collection:
    - Summaries have metadata: {"type": "summary", "source": doc_id}
    - Chunks have metadata: {"type": "chunk", "source": doc_id, "chunk_index": i}
    
    Retrieval happens in two stages:
    1. Coarse retrieval: Search for summaries to identify relevant documents
    2. Fine retrieval: For each relevant document, search for its chunks
    """
    
    def __init__(self, vector_store, n_docs=3, n_chunks_per_doc=5):
        """
        Args:
            vector_store: A PGVector store containing both summaries and chunks
            n_docs: Number of documents to retrieve in coarse retrieval
            n_chunks_per_doc: Number of chunks to retrieve per document in fine retrieval
        """
        self.vector_store = vector_store
        self.n_docs = n_docs
        self.n_chunks_per_doc = n_chunks_per_doc
    
    def get_relevant_documents(self, query: str):
        """
        Perform hierarchical retrieval.
        
        Args:
            query: The search query
            
        Returns:
            list: List of (Document, score) tuples for relevant chunks
        """
        # Step 1: Coarse retrieval - search only summaries
        summaries = self.vector_store.similarity_search_with_score(
            query,
            k=self.n_docs,
            filter={"type": "summary"}
        )
        
        relevant_chunks = []
        
        # Step 2: Fine retrieval - for each relevant document, get its chunks
        for summary_doc, summary_score in summaries:
            doc_id = summary_doc.metadata.get("source")
            
            if doc_id:
                # Search for chunks from this specific document
                chunks = self.vector_store.similarity_search_with_score(
                    query,
                    k=self.n_chunks_per_doc,
                    filter={"source": doc_id, "type": "chunk"}
                )
                
                # Add summary score as context (optional)
                for chunk_doc, chunk_score in chunks:
                    chunk_doc.metadata["parent_summary_score"] = summary_score
                    relevant_chunks.append((chunk_doc, chunk_score))
        
        return relevant_chunks
    
    def get_relevant_documents_simple(self, query: str):
        """
        Simplified version that returns just the documents without scores.
        
        Args:
            query: The search query
            
        Returns:
            list: List of Document objects
        """
        chunks_with_scores = self.get_relevant_documents(query)
        return [doc for doc, score in chunks_with_scores]
