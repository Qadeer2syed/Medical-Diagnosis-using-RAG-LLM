import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import numpy as np
import json
from config import FAISS_INDEX_PMC_PATH, FAISS_INDEX_PUBMED_PATH, PMC_CLEAN_PATH, PUBMED_CLEAN_PATH

# Load the MedCPT model using Transformers
model_name = "ncbi/MedCPT-Article-Encoder"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load FAISS indices
index_pmc = faiss.read_index(FAISS_INDEX_PMC_PATH)
index_pubmed = faiss.read_index(FAISS_INDEX_PUBMED_PATH)

# Load cleaned documents
with open(PMC_CLEAN_PATH, "r", encoding="utf-8") as f:
    pmc_documents = json.load(f)
with open(PUBMED_CLEAN_PATH, "r", encoding="utf-8") as f:
    pubmed_documents = json.load(f)

def encode_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Extract embeddings using the model
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    
    # Convert to numpy array
    return embeddings.cpu().numpy()

def retrieve_documents(query, top_k=1):
    # Compute query embedding using the new encode_text function
    query_embedding = encode_text(query)
    
    # Retrieve from PMC index
    distances_pmc, indices_pmc = index_pmc.search(query_embedding, top_k)
    if len(indices_pmc) == 0 or len(indices_pmc[0]) == 0:
        print("No results found in PMC index.")
        retrieved_pmc = []
    else:
        retrieved_pmc = [pmc_documents[idx] for idx in indices_pmc[0] if idx < len(pmc_documents)]

    # Retrieve from PubMed index
    distances_pubmed, indices_pubmed = index_pubmed.search(query_embedding, top_k)
    if len(indices_pubmed) == 0 or len(indices_pubmed[0]) == 0:
        print("No results found in PubMed index.")
        retrieved_pubmed = []
    else:
        retrieved_pubmed = [pubmed_documents[idx] for idx in indices_pubmed[0] if idx < len(pubmed_documents)]

    return retrieved_pmc, retrieved_pubmed

# For testing:
if __name__ == "__main__":
    test_query = "What are the symptoms of long COVID?"
    pmc_docs, pubmed_docs = retrieve_documents(test_query)
    print("PMC Documents:")
    for doc in pmc_docs:
        print(f"Title: {doc['title']}\nAbstract (truncated): {doc['abstract'][:200]}...\n")
    print("PubMed Documents:")
    for doc in pubmed_docs:
        print(f"Title: {doc['title']}\nAbstract (truncated): {doc['abstract'][:200]}...\n")
