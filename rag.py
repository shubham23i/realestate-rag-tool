from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import shutil
import time
import os

# Modern imports (no deprecation warnings)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    """Initialize LLM and vector store"""
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    """Scrape data from URLs and store in Chroma vector DB"""
    global vector_store
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"

    # Safely close Chroma DB before deleting directory
    try:
        if vector_store and hasattr(vector_store, "_client"):
            vector_store._client.close()
            time.sleep(1)
    except Exception as e:
        print(f"Warning: could not close vector store cleanly: {e}")

    # Now safely delete directory
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
        time.sleep(0.5)

    ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=ef,
        persist_directory=str(VECTORSTORE_DIR)
    )

    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Adding chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    #vector_store.persist()

    yield "Done adding docs to vector database...✅"


def generate_answer(query):
    """Perform retrieval-augmented QA using modern LangChain syntax"""
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    You are a financial assistant.
    Use the following context to answer the question accurately.
    If the answer isn't found, say you don't know.

    Context:
    {context}

    Question:
    {question}

    Provide a concise and factual answer.
    """)

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    response = rag_chain.invoke(query)
    return response.content


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
    ]

    for status in process_urls(urls):
        print(status)

    print("\nGenerating answer...")
    answer = generate_answer("Tell me what was the 30-year fixed mortgage rate along with the date?")
    print(f"\nAnswer:\n{answer}")
