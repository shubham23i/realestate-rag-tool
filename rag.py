from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import shutil
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# -------------------- CONFIG --------------------
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# -------------------- GLOBALS --------------------
llm = None
vector_store = None


# -------------------- INITIALIZATION --------------------
def initialize_llm():
    """Initialize Groq LLM"""
    global llm
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500,
        )


def reset_vector_store():
    """Safely delete existing Chroma DB"""
    global vector_store

    try:
        if vector_store and hasattr(vector_store, "_client"):
            vector_store._client.close()
            time.sleep(1)
    except Exception as e:
        print(f"Warning closing vector store: {e}")

    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
        time.sleep(0.5)

    vector_store = None


def create_vector_store():
    """Create Chroma vector store with HuggingFace embeddings"""
    global vector_store

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


# -------------------- INGESTION --------------------
def process_urls(urls):
    """Load URLs, chunk text, and store embeddings"""
    global vector_store

    yield "🔧 Initializing LLM"
    initialize_llm()

    yield "♻️ Resetting vector store"
    reset_vector_store()

    yield "📦 Creating vector store"
    create_vector_store()

    yield "🌐 Loading URLs"
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    yield "✂️ Splitting text into chunks"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    yield "🧠 Adding documents to vector DB"
    ids = [str(uuid4()) for _ in chunks]
    vector_store.add_documents(chunks, ids=ids)

    yield "✅ Vector store ready"


# -------------------- RAG QA --------------------
def generate_answer(question: str) -> str:
    """Answer questions using RAG"""
    if not vector_store:
        raise RuntimeError("Vector store not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are a financial assistant.
Use ONLY the context provided to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer concisely and factually.
""")

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    response = chain.invoke(question)
    return response.content


# -------------------- LOCAL TEST --------------------
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
    ]

    for msg in process_urls(urls):
        print(msg)

    print("\n🔍 Asking question...")
    answer = generate_answer(
        "What was the 30-year fixed mortgage rate and on what date?"
    )
    print("\nAnswer:\n", answer)
