from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import streamlit as st
import os

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"

store = {}

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.environ.get("GROQ_API_KEY", "")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def build_chain(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatGroq(model_name=GROQ_MODEL, groq_api_key=get_groq_key())

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions about a document.
Use the context below to answer. If the exact answer is not stated,
reason from the context and give the best possible answer.
Do not say you don't know if there is relevant information available.

Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def retrieve_and_format(input_dict):
        question = input_dict["input"]
        retrieved = retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in retrieved)

    chain = (
        {
            "context": RunnableLambda(retrieve_and_format),
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", []))
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
