import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="NSU SEPS Information Chatbot", layout="wide")
load_dotenv()

PDF_STORAGE_DIR = r"/Users/jonayedhossain/Desktop/Chatbot/pdf"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)


# ✅ Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# ✅ Function to extract text from a webpage URL
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator='\n')
        cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return cleaned_text
    except Exception as e:
        st.error(f"Error fetching URL content: {e}")
        return ""


# ✅ Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


# ✅ Create vector store using FAISS
def get_vectorstore(text_chunks):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(text_chunks)
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        vectorstore = FAISS.from_documents(
            documents, SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


# ✅ Build conversation chain using Ollama (LLaMA2)
def get_conversation_chain(vectorstore, model_name="llama2"):
    try:
        llm = Ollama(model=model_name)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None


# ✅ Handle user input and generate response
def handle_userinput(user_question):
    try:
        if 'conversation' not in st.session_state or not st.session_state.conversation:
            st.warning("⚠️ Please upload and process a PDF or URL first.")
            return None

        response = st.session_state.conversation.invoke({"question": user_question})
        return response["answer"]
    except Exception as e:
        st.error(f"Error processing request: {e}")
        return None


# ✅ Load previously saved PDF files
def load_saved_pdfs():
    return [os.path.join(PDF_STORAGE_DIR, f) for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pdf")]


# ✅ Process PDFs
def process_pdfs(pdf_files):
    try:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)

        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("✅ PDFs processed successfully!")
        else:
            st.error("❌ Error: Could not create vector store.")
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")


# ✅ Process text from URL
def process_url_text(url):
    try:
        url_text = get_text_from_url(url)
        if not url_text:
            st.error("❌ No text found in URL.")
            return
        text_chunks = get_text_chunks(url_text)
        vectorstore = get_vectorstore(text_chunks)

        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("✅ URL processed successfully!")
        else:
            st.error("❌ Error creating vector store from URL content.")
    except Exception as e:
        st.error(f"Error processing URL: {e}")


# ✅ Main Streamlit app
def main():
    st.title("NSU SEPS Information Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.write("💬 Ask Questions")
    user_question = st.chat_input("Enter your question...")

    if user_question:
        response = handle_userinput(user_question)
        if response:
            st.session_state.chat_history.append({"question": user_question, "answer": response})

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Sidebar: PDF and URL upload options
    saved_pdfs = load_saved_pdfs()
    with st.sidebar:
        st.subheader("📂 Manage Documents")

        # Process PDFs
        if saved_pdfs:
            if st.button("🔄 Process Saved PDFs"):
                with st.spinner("Processing saved PDFs..."):
                    process_pdfs(saved_pdfs)
        else:
            st.warning("⚠️ No saved PDFs found. Please upload files.")

        # Process URL
        st.subheader("🌐 Process URL")
        url_input = st.text_input("Enter a URL (e.g., webpage, blog post)")

        if st.button("Process URL"):
            if url_input:
                with st.spinner("Fetching and processing URL..."):
                    process_url_text(url_input)
            else:
                st.warning("⚠️ Please enter a valid URL.")


if __name__ == "__main__":
    main()
