class HierarchicalRetriever:
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
