import os
import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# ========= CONFIG ==========
POPPLER_PATH = r"C:\path\to\poppler\bin"  # ← তোমার Poppler এর path এখানে দাও
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

st.set_page_config(page_title="NSU SEPS Chatbot", layout="wide")
load_dotenv()

# ========= 1. GitHub থেকে PDF ডাউনলোড ==========
def download_pdf_from_github(raw_url, save_folder=PDF_FOLDER):
    try:
        filename = raw_url.split("/")[-1]
        save_path = os.path.join(save_folder, filename)

        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            st.success(f"✅ Downloaded: {filename}")
        else:
            st.error(f"❌ Failed to download: Status code {response.status_code}")
    except Exception as e:
        st.error(f"Error downloading PDF: {e}")

# ========= 2. Text Extraction ==========
def extract_text_with_pypdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.warning(f"PyPDF2 failed on {os.path.basename(pdf_path)}: {e}")
    return text

def extract_text_with_ocr(pdf_path):
    text = ""
    try:
        pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        for page in pages:
            page_text = pytesseract.image_to_string(page, config='--psm 6')
            text += page_text + "\n"
    except Exception as e:
        st.error(f"OCR failed on {os.path.basename(pdf_path)}: {e}")
    return text

def get_pdf_text():
    full_text = ""
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        st.error("❌ No PDFs found.")
        return ""

    for filename in pdf_files:
        path = os.path.join(PDF_FOLDER, filename)
        text = extract_text_with_pypdf(path)
        if not text.strip():
            st.info(f"Trying OCR for {filename}...")
            text = extract_text_with_ocr(path)
        if not text.strip():
            st.warning(f"No text extracted from {filename}")
        full_text += text + "\n"
    return full_text

# ========= 3. LangChain Setup ==========
def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    try:
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        return FAISS.from_documents(docs, SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2'))
    except Exception as e:
        st.error(f"Vectorstore Error: {e}")
        return None

def get_conversation_chain(vectorstore, model_name="mistral"):
    try:
        llm = Ollama(model=model_name)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    except Exception as e:
        st.error(f"Chain Error: {e}")
        return None

# ========= 4. Process All PDFs ==========
def process_all_pdfs():
    raw_text = get_pdf_text()
    if not raw_text.strip():
        st.error("No valid text found in PDFs.")
        return

    chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(chunks)
    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("✅ PDFs processed.")
    else:
        st.error("Failed to create vectorstore.")

# ========= 5. Handle Question ==========
def handle_userinput(user_question):
    try:
        if 'conversation' not in st.session_state or not st.session_state.conversation:
            st.warning("Process PDFs first.")
            return None
        response = st.session_state.conversation.invoke({"question": user_question})
        return response["answer"]
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ========= 6. Streamlit UI ==========
def main():
    st.title("NSU SEPS Info Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("📥 Download PDF from GitHub")
        github_raw_url = st.text_input("Enter GitHub Raw PDF URL")
        if st.button("Download PDF"):
            if github_raw_url:
                download_pdf_from_github(github_raw_url)
            else:
                st.warning("Please enter a valid raw GitHub URL.")

        st.subheader("📂 Process PDFs")
        if st.button("🔄 Process All PDFs"):
            with st.spinner("Processing..."):
                process_all_pdfs()

    st.write("💬 Ask a question:")
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

if __name__ == "__main__":
    main()
