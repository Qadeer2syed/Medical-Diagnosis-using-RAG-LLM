# scripts/index_documents.py

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import random
import sys
import os
import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from config import PMC_CLEAN_PATH, PUBMED_CLEAN_PATH, FAISS_INDEX_PMC_PATH, FAISS_INDEX_PUBMED_PATH

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# Load ncbi model and tokenizer
MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def embed_text_batch(texts, batch_size=16):
    """Embed a list of texts using batching."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.extend(embeddings)

    return all_embeddings


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks of tokens."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokenizer.convert_tokens_to_string(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def create_index(json_path, index_path, text_field="abstract", chunk_size=500, overlap=30):
    with open(json_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # ✅ Randomly sample 60% of documents
    sample_size = int(0.6 * len(documents))
    documents = random.sample(documents, sample_size)

    vectors = []
    metadata = []

    for doc in documents:
        text = doc.get(text_field, "")
        if not text:
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue

        # ✅ Batched embedding for speed
        chunk_embeddings = embed_text_batch(chunks, batch_size=16)
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            vectors.append(embedding)
            metadata.append({
                "original_doc": doc,
                "chunk_index": i,
                "chunk_text": chunk
            })

    vectors_np = np.stack(vectors)
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_np)

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")

    # Save metadata
    metadata_path = index_path + "_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return metadata

# Create index for PMC Patients
create_index(PMC_CLEAN_PATH, FAISS_INDEX_PMC_PATH, text_field="abstract")

# Create index for PubMed Articles
create_index(PUBMED_CLEAN_PATH, FAISS_INDEX_PUBMED_PATH, text_field="abstract")
