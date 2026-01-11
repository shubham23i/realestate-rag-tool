import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_llm():
    global llm

    if llm is None:
        llm = ChatGroq(
            api_key=os.environ["GROQ_API_KEY"],
            model=os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile"),
            temperature=0.3,
            max_tokens=512,
        )


def process_urls(urls):
    global vector_store

    yield "Initializing model..."
    initialize_llm()

    yield "Loading URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    yield "Splitting text..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=150,
    )
    split_docs = splitter.split_documents(docs)

    yield "Creating embeddings..."
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        ),
    )

    vector_store.add_documents(split_docs)

    yield "✅ Done! You can now ask questions."


def generate_answer(question: str) -> str:
    if vector_store is None:
        raise RuntimeError("Please process URLs first.")

    initialize_llm()

    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    messages = [
        SystemMessage(
            content=(
                "You are a real estate research assistant.\n"
                "Answer ONLY from the provided context.\n"
                "If the answer is not present, say 'I don't know.'"
            )
        ),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{question}"
        ),
    ]

    response = llm.invoke(messages)
    return response.content
