# 📄 PDF Chatbot — RAG with LangChain + Groq

Chat with any PDF using Retrieval Augmented Generation (RAG).
Upload a PDF, ask questions, get accurate answers grounded in the document.

## 🔍 How It Works

PDF → PyPDF loader → text chunks (500 tokens) → FAISS vector store
→ retriever (top 6 chunks) → Groq LLaMA 3.1 8B → answer

## 🛠️ Stack

| Component | Tool |
|---|---|
| Framework | Streamlit |
| RAG Pipeline | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Memory | LangChain ChatMessageHistory |

## ✨ Features

- Upload any PDF via sidebar
- Conversational memory — remembers previous questions
- Context-aware answers from document chunks
- Fast inference via Groq API (free tier)

## 🚀 Run Locally

```bash
git clone https://github.com/Git169-hub/pdf-chatbot
cd pdf-chatbot
pip install -r requirements.txt

# Add your Groq API key
export GROQ_API_KEY=your_key_here

streamlit run app.py
```

Get a free Groq API key at: https://console.groq.com

## 📁 Project Structure

pdf-chatbot/
├── app.py           # Streamlit UI + session management
├── rag_pipeline.py  # RAG chain: load → chunk → embed → retrieve → LLM
└── requirements.txt


## 💡 RAG Pipeline Details

- **Chunking:** RecursiveCharacterTextSplitter — 500 tokens, 50 overlap
- **Embeddings:** HuggingFace MiniLM-L6-v2 (runs locally, no API cost)
- **Retrieval:** FAISS similarity search, top 6 chunks per query
- **LLM:** LLaMA 3.1 8B via Groq (fast, free tier available)
- **Memory:** Per-session chat history via LangChain RunnableWithMessageHistory








