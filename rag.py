import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

CHUNK_SIZE = 800
MAX_CONTEXT_CHARS = 6000   # 🔑 critical
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_llm():
    global llm
    if llm is None:
        llm = ChatGroq(
            api_key=os.environ["GROQ_API_KEY"],
            model="llama3-8b-8192",   # ✅ SAFE MODEL
            temperature=0.2,
            max_tokens=400,
        )


def process_urls(urls):
    global vector_store

    yield "Initializing LLM..."
    initialize_llm()

    yield "Loading URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    yield "Splitting text..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(docs)

    yield "Creating vector store..."
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        ),
    )

    vector_store.add_documents(split_docs)
    yield "✅ URLs processed. Ask questions now."


def generate_answer(question: str) -> str:
    if vector_store is None:
        raise RuntimeError("Please process URLs first.")

    initialize_llm()

    # Search for relevant docs
    docs = vector_store.similarity_search(question, k=4)

    # Build context safely
    context = ""
    for d in docs:
        if len(context) + len(d.page_content) > MAX_CONTEXT_CHARS:
            break
        context += d.page_content.strip() + "\n\n"

    # Build messages as plain dicts
    messages = [
        {"role": "system", "content": (
            "You are a real estate research assistant. "
            "Answer only using the provided context."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]

    # Call Groq
    response = llm.invoke(messages)

    # Some versions of Groq return response differently
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict) and "content" in response:
        return response["content"]
    else:
        return str(response)
