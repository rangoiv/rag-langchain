import os
from typing import List

from datasets import load_dataset
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEndpoint
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


def get_documents(env) -> list[str]:
    """
    Returns a list of max_documents MSMarco documents cut off at position max_characters
    """
    dataset = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        split="train",
        streaming=True,
        token=env["hugging_face.api_key"],
    )
    documents = []
    max_documents = env["dataset.max_documents"]
    progress_bar = tqdm(range(max_documents), desc="Fetching documents")
    # Try more times if document is not well formated.
    for _, doc in zip(range(2 * max_documents) + 100, dataset):
        if len(documents) == max_documents:
            break
        try:
            passage = doc["passages"]["passage_text"][
                doc["passages"]["is_selected"].index(1)
            ]
            documents.append(passage[: env["dataset.max_characters"]])
            progress_bar.update()
        except Exception:
            continue
    else:
        print(f"Fetched all possible documents ({len(documents)})")
    return documents


def create_index_in_not_exist(env):
    """
    Create index if it doesn't exist, returns True if created
    """
    client = Pinecone(api_key=env["pinecone.api_key"])

    index_name = env["pinecone.index_name"]

    if not client.has_index(index_name):
        client.create_index(
            name=index_name,
            dimension=env["pinecone.dimension"],
            spec=ServerlessSpec(cloud="aws", region=env["pinecone.environment"]),
        )
        return True
    return False


def get_vector_store(env):
    """
    Returns PineconeVectorStore instance created from environment
    """
    pinecone_embeddings = PineconeEmbeddings(
        model=env["pinecone.embeddings_model"],
        pinecone_api_key=env["pinecone.api_key"],
        document_params=dict(input_type="passage"),
        query_params=dict(input_type="query"),
        dimension=env["pinecone.dimension"],
        batch_size=1,
    )
    # Needed for creating vector store
    os.environ.setdefault("PINECONE_API_KEY", env["pinecone.api_key"])
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=env["pinecone.index_name"], embedding=pinecone_embeddings
    )
    return vector_store


def get_llm(env):
    """
    Returns BaseLanguageModel instance instantiated from Hugging face
    """
    llm = HuggingFaceEndpoint(
        repo_id=env["llm.model_name"],
        temperature=env["llm.temperature"],
        max_new_tokens=env["llm.max_new_tokens"],
        huggingfacehub_api_token=env["hugging_face.api_key"],
    )
    return llm


def _shorten_string(text: str, llm: BaseLanguageModel, max_tokens: int):
    """
    Shortens text to approximately max_token length
    Aproximate length is calculated based on character length and current token length,
    and resulting text token length is aproximately of size max_token
    """
    token_length = llm.get_num_tokens(text)
    char_length = len(text)
    new_length = int(char_length * max_tokens / token_length)
    return text[:new_length]


def _create_stuff_shortened_documents_chain(
    llm: BaseLanguageModel, prompt: BasePromptTemplate, max_tokens: int
):
    def shorten_documents(input: dict):
        documents: List[Document] = input["context"]
        for document in documents:
            document.page_content = _shorten_string(
                document.page_content, llm, max_tokens=max_tokens
            )
        input["context"] = documents
        return input

    return shorten_documents | create_stuff_documents_chain(llm, prompt)


def get_rag_chain(env, vector_store: VectorStore, llm: BaseLanguageModel):
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt_template = (
        "<|system|>\n"
        f"{system_prompt}"
        "<|end|>\n"
        "<|user|>\n"
        "{input}<|end|>\n"
        "<|assistant|>\n"
    )
    prompt = PromptTemplate.from_template(prompt_template)

    retriever = vector_store.as_retriever(search_kwargs=dict(k=env["rag.documents"]))

    answer_chain = _create_stuff_shortened_documents_chain(
        llm, prompt, env["rag.max_tokens_per_document"]
    )

    chain = create_retrieval_chain(retriever, answer_chain)

    return chain
