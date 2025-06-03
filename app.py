import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from PyPDF2 import PdfReader

# ========== CONFIG ==========
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)
load_dotenv()
st.set_page_config(page_title="NSU SEPS Chatbot", layout="wide")

# ========== PDF TEXT EXTRACTION ==========
def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.warning(f"Failed to read {path}: {e}")
        return ""

# ========== TEXT SPLIT & VECTOR STORE ==========
def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    try:
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        return FAISS.from_documents(docs, SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2'))
    except Exception as e:
        st.error(f"Vector store error: {e}")
        return None

# ========== CONVERSATION CHAIN ==========
def get_conversation_chain(vectorstore, model_name):
    try:
        llm = Ollama(model=model_name)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    except Exception as e:
        st.error(f"Chain build failed: {e}")
        return None

# ========== PROCESS PDF FILES ==========
def process_uploaded_pdfs(files, model_name):
    try:
        combined_text = ""
        for file in files:
            file_path = os.path.join(PDF_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            text = extract_text_from_pdf(file_path)
            st.write(f"📄 {file.name}: {len(text)} characters extracted")
            combined_text += text + "\n"

        if not combined_text.strip():
            st.error("No text extracted.")
            return

        chunks = get_text_chunks(combined_text)
        st.write(f"🧩 Total chunks: {len(chunks)}")

        vectorstore = get_vectorstore(chunks)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore, model_name)
            st.success("✅ PDF processing complete.")
        else:
            st.error("❌ Failed to create vectorstore.")
    except Exception as e:
        st.error(f"PDF processing error: {e}")

# ========== HANDLE USER INPUT ==========
def handle_user_input(question):
    try:
        if not st.session_state.get("conversation"):
            st.warning("Please process a PDF first.")
            return None
        response = st.session_state.conversation.invoke({"question": question})
        return response.get("answer", "")
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ========== MAIN STREAMLIT APP ==========
def main():
    st.title("📘 NSU SEPS Information Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("📂 Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        st.subheader("🤖 Select Model")
        model_name = st.selectbox("Ollama model", ["mistral", "llama2", "gemma", "phi"])

        if st.button("🔄 Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    process_uploaded_pdfs(uploaded_files, model_name)
            else:
                st.warning("Upload at least one PDF.")

    # Chat section
    st.subheader("💬 Ask a question")
    user_question = st.chat_input("Type your question here...")
    if user_question:
        answer = handle_user_input(user_question)
        if answer:
            st.session_state.chat_history.append({"question": user_question, "answer": answer})

    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])

if __name__ == "__main__":
    main()
