import os
from typing import List

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# =========================
# Configuration
# =========================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"

# =========================
# Globals (Streamlit-safe)
# =========================

llm = None
vector_store = None

# =========================
# Initialization
# =========================

def init_llm():
    global llm

    if llm is not None:
        return

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found")

    llm = ChatGroq(
        model=model,
        temperature=0,
        max_tokens=512,
    )

# =========================
# URL Processing
# =========================

def process_urls(urls: List[str]):
    global vector_store

    yield "🔧 Initializing model..."
    init_llm()

    yield "🌐 Loading URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    yield "✂️ Splitting text..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    yield "🧠 Creating embeddings..."
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    vector_store.add_documents(chunks)

    yield "✅ Done! You can now ask questions."

# =========================
# Question Answering
# =========================

def generate_answer(question: str) -> str:
    if vector_store is None:
        return "⚠️ Please enter URLs and click **Process URLs** first."

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful real estate research assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

<context>
{context}
</context>

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    result = retrieval_chain.invoke({"input": question})
    return result["answer"]
