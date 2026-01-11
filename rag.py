import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load env
load_dotenv()

CHUNK_SIZE = 1000
COLLECTION_NAME = "real_estate"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm = None
vector_store = None


def initialize_llm():
    global llm

    if llm is None:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.3,
            max_tokens=500,
        )


def process_urls(urls):
    global vector_store

    initialize_llm()

    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    vector_store.add_documents(docs)

    return "URLs processed successfully."


def generate_answer(question: str):
    if vector_store is None:
        raise RuntimeError("Please process URLs first.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful real estate assistant.
        Use ONLY the context below to answer the question.
        If the answer is not in the context, say "I don't know".

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

    response = chain.invoke(question)
    return response.content
