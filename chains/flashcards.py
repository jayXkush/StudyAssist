import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 🔴 REQUIRED
load_dotenv()

DB_PATH = "./data/chroma"


# ---------- SHARED HELPERS ----------

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="learning_assistant"
    )


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------- SUMMARY CHAIN ----------

def get_summary_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """You are a helpful study assistant.
Using ONLY the context below, write a clear and concise summary about "{question}".
If the topic is not covered, say so clearly.

Context:
{context}

Summary:"""
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def generate_summary(topic: str, vectorstore) -> str:
    try:
        chain = get_summary_chain(vectorstore)
        return chain.invoke(topic)
    except Exception as e:
        return f"❌ Error generating summary: {e}"


# ---------- FLASHCARD CHAIN ----------

def get_flashcard_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """You are a learning assistant.
Using ONLY the context below, create 5 flashcards about "{question}".

Rules:
- Each flashcard must be concise
- Answers must come strictly from the context
- If information is missing, skip that flashcard

Context:
{context}

Format:
1. Front: ...
   Back: ...
2. Front: ...
   Back: ...

Flashcards:"""
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def generate_flashcards(topic: str, vectorstore) -> str:
    try:
        chain = get_flashcard_chain(vectorstore)
        return chain.invoke(topic)
    except Exception as e:
        return f"❌ Error generating flashcards: {e}"
