import os
from openai import AzureOpenAI
import project_config
import faiss
import pickle
from project_app.embedding import get_embedding, load_faiss_index, load_metadata
import numpy as np


azure_endpoint = project_config.AZURE_END_POINT
azure_api_key = project_config.AZURE_OPENAI_KEY
model_name = "gpt-4o"
deployment_name = "gpt-4o"
api_version = "2024-12-01-preview"
embedding_model_name = "text-embedding-ada-002"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
)


def get_chat_response(user_input):
    response = client.chat.completions.create(
            messages=[
                
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            max_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            model=deployment_name
        )
    reply = response.choices[0].message.content
    return reply

# ================================
# Load FAISS and Metadata at Startup
# ================================

print("[INFO] Loading FAISS index and metadata...")
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"

index = load_faiss_index(INDEX_PATH)
metadata = load_metadata(METADATA_PATH)
print(f"[INFO] Index loaded with {index.ntotal} vectors.")
print(f"[INFO] Metadata loaded with {len(metadata)} entries.")


# ================================
# Helper: Retrieve Relevant Context
# ================================
def retrieve_context(user_query, top_k=3):
    # 1. Embed user query
    emb = get_embedding(user_query)
    emb_vec = np.array(emb, dtype='float32').reshape(1, -1)

    # 2. Search FAISS
    D, I = index.search(emb_vec, top_k)
    retrieved_chunks = [metadata[idx]['text'] for idx in I[0]]

    # 3. Combine into context string
    context = "\n\n---\n\n".join(retrieved_chunks)
    return context