import streamlit as st
import tempfile, os
from rag_pipeline import build_chain

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and "chain" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        with st.spinner("Processing PDF..."):
            st.session_state.chain = build_chain(tmp_path)
            st.session_state.history = []
        st.success("Ready. Ask your questions.")

if "chain" in st.session_state:
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    user_input = st.chat_input("Ask something about the PDF...")
    if user_input:
        st.chat_message("user").write(user_input)

        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}}
            )

        st.chat_message("assistant").write(answer)
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("assistant", answer))
else:
    st.info("Upload a PDF from the sidebar to start.")
