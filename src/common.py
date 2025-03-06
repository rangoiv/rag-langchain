import os
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings


def get_vector_store(env):
    """
    Returns PineconeVectorStore instance created from environment
    """
    pinecone_embeddings = PineconeEmbeddings(
        model=env["pinecone.embeddings_model"],
        pinecone_api_key=env["pinecone.api_key"],
        document_params=dict(input_type="passage"),
        batch_size=1,
    )
    # Needed for creating vector store
    os.environ.setdefault("PINECONE_API_KEY", env["pinecone.api_key"])
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=env["pinecone.index_name"], embedding=pinecone_embeddings
    )
    return vector_store
