from datasets import load_dataset
from envyaml import EnvYAML

from common import get_vector_store


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
