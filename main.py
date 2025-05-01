import logging
import os
import faiss
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Application started.")

# --- Load Embedding Model ---
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def load_pdf_text(uploaded_files):
    """Extract text from uploaded PDFs."""
    pdf_texts = {}
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        text = "".join(
            [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        )
        pdf_texts[uploaded_file.name] = text
    return pdf_texts

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into overlapping chunks."""
    text_len = len(text)
    return [text[i : i + chunk_size] for i in range(0, text_len, chunk_size - chunk_overlap)]

def embed_text_chunks(text_chunks):
    """Generate embeddings for text chunks."""
    return embedding_model.encode(text_chunks, convert_to_numpy=True)

def build_faiss_index(embeddings):
    """Build FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, top_k=10):
    """Search FAISS index."""
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    return I[0]

def generate_answer_groq(query, context_chunks):
    """Generate response using Groq API."""
    if not GROQ_API_KEY:
        return "Please set your Groq API key."
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GENERATION_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Context: {' '.join(context_chunks)}\nQuestion: {query}\nAnswer:"},
        ],
        "temperature": 0.1,
        "max_tokens": 800,
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# --- Streamlit UI ---
st.title("Ask Your PDF: AI Q&A App")
st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    pdf_texts = load_pdf_text(uploaded_files)
    all_chunks = [chunk_text(text) for text in pdf_texts.values()]
    all_chunks = [item for sublist in all_chunks for item in sublist]
    embeddings = embed_text_chunks(all_chunks)
    faiss_index = build_faiss_index(embeddings)
    
    st.sidebar.success("PDFs processed successfully!")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        query_embedding = embed_text_chunks([user_question])
        indices = search_faiss_index(faiss_index, np.array(query_embedding, dtype=np.float32))
        context_chunks = [all_chunks[idx] for idx in indices]
        answer = generate_answer_groq(user_question, context_chunks)
        st.subheader("Generated Answer:")
        st.write(answer)