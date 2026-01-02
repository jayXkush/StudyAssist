# AI Learning Assistant: RAG-based Study Companion

## Overview
A Streamlit-powered RAG (Retrieval-Augmented Generation) application that helps you study PDF documents through intelligent conversations, summaries, and flashcard generation. Upload your study materials and chat with an AI tutor that understands your content.

## 🚀 Features

- **💬 Interactive Chat**: Ask questions about your uploaded PDF documents and get context-aware answers
- **📝 Topic Summaries**: Generate focused summaries on specific topics from your documents
- **🃏 Flashcard Generation**: Create study flashcards for any topic in your documents
- **📚 PDF Processing**: Upload and process PDF documents with intelligent text chunking
- **🔍 Semantic Search**: Find relevant information using vector embeddings

## 🏗️ Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│  Streamlit UI   │◄───►│  LangChain        │◄───►│  Groq LLM API       │
│  (Frontend)     │     │  (RAG Pipeline)   │     │  (LLaMA 3.1)        │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
                                 ▲
                                 │
                         ┌───────┴───────┐
                         │  ChromaDB      │
                         │  (Vector Store)│
                         └─────────────────┘
```

### Components

1. **Streamlit Frontend**
   - Web-based user interface with tabs for different features
   - File upload for PDF documents
   - Chat interface, summary generator, and flashcard creator

2. **RAG Pipeline (LangChain)**
   - Document processing and text splitting
   - Vector embedding using sentence transformers
   - Retrieval-augmented generation for context-aware responses

3. **Vector Store (ChromaDB)**
   - In-memory storage of document embeddings
   - Semantic similarity search for relevant context

4. **LLM Integration (Groq)**
   - Uses LLaMA 3.1 8B model via Groq API
   - Fast inference for real-time responses

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.9+**: Primary programming language
- **Streamlit**: Web application framework
- **LangChain**: RAG application framework
- **ChromaDB**: Vector database for document storage
- **Groq**: LLM API service (LLaMA 3.1)

### AI/ML Libraries
- **sentence-transformers**: Document embeddings (all-MiniLM-L6-v2)
- **transformers**: Hugging Face transformers library
- **torch**: PyTorch for deep learning
- **accelerate**: Model optimization

### Document Processing
- **pypdf**: PDF text extraction
- **langchain-community**: Community LangChain components
- **langchain-huggingface**: Hugging Face integrations
- **langchain-chroma**: ChromaDB integration
- **langchain-groq**: Groq API integration

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git
- Groq API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jayXkush/StudyAssist.git
   cd StudyAssist
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── chains/               # LangChain chains
│   ├── chat.py          # Q&A chat chain
│   └── flashcards.py    # Summary and flashcard generation
├── utils/                # Utility functions
│   └── ingestion.py     # PDF processing and vectorization
└── data/                # Data storage (auto-created)
    ├── uploads/         # Uploaded PDFs
    └── chroma/          # ChromaDB storage
```

## 🎯 How to Use

1. **Upload a PDF**: Use the sidebar to upload a PDF document and click "Process Document"
2. **Chat with AI Tutor**: In the "Tutor Chat" tab, ask questions about your document
3. **Generate Summaries**: In the "Topic Summary" tab, enter a topic to get a focused summary
4. **Create Flashcards**: In the "Flashcards" tab, generate study flashcards for any topic

## 🔧 Configuration

- **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: LLaMA 3.1 8B Instant via Groq API
- **Chunk Size**: 1000 characters with 200 character overlap
- **Retrieval**: Top 3 most relevant chunks for context

## 🐳 Docker Deployment

A `Dockerfile` is included for containerized deployment:

```bash
docker build -t study-assistant .
docker run -p 8501:8501 study-assistant
```

## 🌐 Deployment

The app is configured for easy deployment on Streamlit Cloud with the included `Procfile` and `packages.txt`.

