import warnings

from common import (
    create_index_in_not_exist,
    get_documents,
    get_llm,
    get_rag_chain,
    get_vector_store,
)
from envyaml import EnvYAML
from huggingface_hub.utils import HfHubHTTPError
from langchain_core.runnables import Runnable
from prompt_toolkit import PromptSession


def invoke(chain: Runnable, query: str) -> str | None:
    try:
        answer = chain.invoke({"input": query})
        return answer["answer"]
    except HfHubHTTPError as e:
        if 500 <= e.response.status_code < 600:
            print(f"{e}. Try again in a few minutes.")
        else:
            raise e


def create_index_and_chain(env):
    print("Connecting to vector store")
    created = create_index_in_not_exist(env)
    vector_store = get_vector_store(env)
    if created:
        print("Created index")
        documents = get_documents(env)
        print("Adding documents to vector store")
        vector_store.add_texts(documents)
    print("Connecting to llm")
    llm = get_llm(env)
    chain = get_rag_chain(env, vector_store, llm)
    print("Ready!")
    return chain


def main():
    env = EnvYAML("env.yaml")
    chain = create_index_and_chain(env)
    print(
        "The bot will answer questions based on MSMarco dataset. Type 'exit' or 'q' to exit."
    )
    print("Try with questions based on MSMarco: What is RBA?")
    print()

    prompt_session = PromptSession()
    while True:
        try:
            query = prompt_session.prompt("Question: ")
        except (EOFError, KeyboardInterrupt):
            break  # Handle Ctrl+D or Ctrl+C to exit
        if query in ["exit", "q"]:
            break
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            answer = invoke(chain, query)
        if answer is not None:
            print("Answer:", answer)
            print()


if __name__ == "__main__":
    main()
