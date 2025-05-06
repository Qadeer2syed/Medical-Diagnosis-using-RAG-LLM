# config.py



MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#MODEL_NAME = "medalpaca/medalpaca-7b"
#MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"


# You can choose one of the following:
# Option A (biomedical, if available):#
#EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_MODEL = "ncbi/MedCPT-Article-Encoder"
# Option B (general-purpose):
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

#FAISS_INDEX_PMC_PATH = "data/faiss_index_pmc1"
#FAISS_INDEX_PUBMED_PATH = "data/faiss_index_pubmed1"

FAISS_INDEX_PMC_PATH = "data/faiss_index_pmc_MedCPT"
FAISS_INDEX_PUBMED_PATH = "data/faiss_index_pubmed1_MedCPT"

PMC_RAW_PATH = "data/PMC_Patients.csv"
PMC_CLEAN_PATH = "data/PMC_Patients_Clean.json"

PUBMED_RAW_PATH = "data/pubmed_articles.csv"
PUBMED_CLEAN_PATH = "data/pubmed_articles_Clean.json"
