import os
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings


def get_vector_store(collection: str = "documents") -> PGVector:
    connection = os.getenv("DATABASE_URL") 
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    vector_store = PGVector(  
        embeddings=OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url),
        collection_name=collection,  
        connection=connection,  
        use_jsonb=True,
    )
    return vector_store
