import os
print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))
print("GROQ_MODEL:", os.getenv("GROQ_MODEL"))

from uuid import uuid4
from dotenv import load_dotenv
import os
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found in environment variables")

    if llm is None:
        llm = ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.9,
            max_tokens=500,
        )


def process_urls(urls):
    global vector_store

    yield "Initializing components..."
    initialize_components()

    yield "Loading data..."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    yield "Creating embeddings..."
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
    )

    vector_store.add_documents(docs)

    yield "Done! You can now ask questions."


def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Please process URLs first.")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a financial assistant.
        Use the context below to answer the question.
        If the answer is not present, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    
    response = chain.invoke({"question": query})
    return response.content
