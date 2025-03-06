
from common import create_index_in_not_exist, get_documents, get_vector_store
from envyaml import EnvYAML


def main():
    env = EnvYAML("env.yaml", env_file=".env")

    # Fetch documents
    documents = get_documents(env)

    # Create vector store
    vector_store = get_vector_store(env)

    # Delete curent index and add new loaded documents
    created = create_index_in_not_exist(env)

    if not created:
        vector_store.delete(delete_all=True)
    vector_store.add_texts(documents)


if __name__ == "__main__":
    main()
