import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = "./data/chroma"
UPLOAD_DIR = "./data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_pdf(uploaded_file):
    # Save file
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in Chroma (in-memory for Streamlit Cloud)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="learning_assistant"
    )

    return vectorstore
