import os
from langchain_postgres import PGVector
from langchain_community.embeddings import OllamaEmbeddings


def get_vector_store():
    connection = os.getenv("DATABASE_URL") 
    collection_name = "documents"
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    vector_store = PGVector(  
        embeddings=OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url),
        collection_name=collection_name,  
        connection=connection,  
        use_jsonb=True,
    )
    return vector_store
