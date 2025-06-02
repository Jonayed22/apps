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
from pdf2image import convert_from_path
import pytesseract

# ========== CONFIG ==========
POPPLER_PATH = r"C:\path\to\poppler\bin"  # <-- Change this to your Poppler bin folder path on Windows
PDF_FOLDER = "pdfs"  # folder containing your PDFs

st.set_page_config(page_title="NSU SEPS Information Chatbot", layout="wide")
load_dotenv()
os.makedirs(PDF_FOLDER, exist_ok=True)

# ========== TEXT EXTRACTION ==========

def extract_text_with_pypdf(pdf_path):
    from PyPDF2 import PdfReader
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
    st.write(f"Found PDFs: {pdf_files}")
    if not pdf_files:
        st.error(f"No PDFs found in folder: {PDF_FOLDER}")
        return ""
    for filename in pdf_files:
        path = os.path.join(PDF_FOLDER, filename)
        text = extract_text_with_pypdf(path)
        if not text.strip():
            st.info(f"No text found in {filename} with PyPDF2, trying OCR...")
            text = extract_text_with_ocr(path)
        if not text.strip():
            st.warning(f"No text extracted from {filename} even with OCR.")
        full_text += text + "\n"
    return full_text

# ========== TEXT PROCESSING & CHATBOT ==========

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50, length_function=len)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    try:
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        return FAISS.from_documents(docs, SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2'))
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversation_chain(vectorstore, model_name="mistral"):
    try:
        llm = Ollama(model=model_name)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def process_all_pdfs():
    try:
        raw_text = get_pdf_text()
        st.write(f"📄 Extracted text length: {len(raw_text)}")
        if len(raw_text.strip()) == 0:
            st.error("❌ No valid text extracted from PDFs. Check files or folder.")
            return
        chunks = get_text_chunks(raw_text)
        st.write(f"🧩 Total chunks: {len(chunks)}")
        if len(chunks) == 0:
            st.error("❌ No text chunks created from extracted text.")
            return
        vectorstore = get_vectorstore(chunks)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("✅ PDFs processed successfully.")
        else:
            st.error("❌ Could not create vector store.")
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")

def handle_userinput(user_question):
    try:
        if 'conversation' not in st.session_state or not st.session_state.conversation:
            st.warning("⚠️ Please click 'Process PDFs' first.")
            return None
        response = st.session_state.conversation.invoke({"question": user_question})
        return response["answer"]
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ========== STREAMLIT UI ==========

def main():
    st.title("NSU SEPS Information Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("📂 Manage PDFs")
        if st.button("🔄 Process PDFs"):
            with st.spinner("Processing PDFs..."):
                process_all_pdfs()

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

if __name__ == "__main__":
    main()
