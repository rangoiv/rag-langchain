import os
from datasets import load_dataset
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from envyaml import EnvYAML


def get_documents(env) -> list[str]:
    """
    Returns a list of max_documents Gutenberg documents cut off at position max_characters
    """
    dataset = load_dataset(
        env["dataset.name"],
        split=env["dataset.split"],
        streaming=True,
        token=env["hugging_face.api_key"],
    )
    documents = []
    for _, doc in zip(range(env["dataset.max_documents"]), dataset):
        text_col = env["dataset.text_col"]
        if text_col in doc:
            documents.append(doc[text_col][: env["dataset.max_characters"]])
    return documents


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


def main():
    env = EnvYAML("env.yaml", env_file=".env")

    # Fetch documents
    documents = get_documents(env)

    # Create vector store
    vector_store = get_vector_store(env)

    # Delete curent index and add new loaded documents
    vector_store.delete(delete_all=True)
    vector_store.add_texts(documents)


if __name__ == "__main__":
    main()
