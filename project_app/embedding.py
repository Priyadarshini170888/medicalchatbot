import os
from openai import AzureOpenAI
import project_config
from PyPDF2 import PdfReader
import numpy as np
import faiss
import pickle

# Azure config
azure_endpoint = project_config.AZURE_END_POINT
azure_api_key = project_config.AZURE_OPENAI_KEY
api_version = "2024-12-01-preview"

chat_model_name = "gpt-4o"
embedding_model_name = "text-embedding-ada-002"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
)

# -------------- UTILS --------------
def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""   
    return text     

def split_text(text, max_tokens=800):
    """
    Splits text into roughly max_tokens-sized chunks.
    Here we just split by words as a simple heuristic.
    For better accuracy use tiktoken or similar libraries.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=embedding_model_name
    )
    return response.data[0].embedding

# -------------- VECTOR STORE UTILS --------------
def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)

def save_metadata(metadata, path):
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)

def load_metadata(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# -------------- MAIN --------------
if __name__ == "__main__":
    pdf_path = os.path.join("Data", "HR-Policies-Manuals.pdf")
    index_path = "faiss_index.bin"
    metadata_path = "metadata.pkl"

    # 1. Extract text
    text = pdf_to_text(pdf_path)
    # print("Extracted Text:", text)

    # 2. Split text into chunks
    chunks = split_text(text, max_tokens=800)
    print(f"Total chunks created: {len(chunks)}")

     # ----- 3. LOAD OR CREATE INDEX -----
    if os.path.exists(index_path):
        index = load_faiss_index(index_path)
        print("Loaded existing FAISS index.")
    else:
        print("Creating new FAISS index.")
        # Assuming 1536-dimensional embeddings (e.g. ada-002)
        index = faiss.IndexFlatL2(1536)

    # ----- 4. LOAD OR CREATE METADATA -----
    if os.path.exists(metadata_path):
        metadata = load_metadata(metadata_path)
    else:
        metadata = []

    # ----- 5. PROCESS EACH CHUNK -----
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}...")
        emb = get_embedding(chunk)
        emb_vector = np.array(emb, dtype='float32').reshape(1, -1)

        # Add to FAISS
        index.add(emb_vector)

        # Add metadata
        metadata.append({
            "pdf": pdf_path,
            "chunk_index": i,
            "text": chunk
        })

    # ----- 6. SAVE INDEX AND METADATA -----
    save_faiss_index(index, index_path)
    save_metadata(metadata, metadata_path)

    print("✅ All embeddings stored in FAISS.")
    print(f"FAISS index now contains {index.ntotal} vectors.")