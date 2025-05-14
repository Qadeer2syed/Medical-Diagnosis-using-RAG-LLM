
# 🧠 Medical Diagnosis using RAG & LLM

A Retrieval-Augmented Generation (RAG) pipeline for clinical question answering and diagnosis generation using open-source Large Language Models (LLMs) and biomedical literature. This system integrates FAISS-based retrieval with transformer-based generation models to simulate expert-level medical reasoning.

## 📁 Project Structure

```
FinalProjectCode/
│
├── config.py                   # Path and model configuration
├── test_scores.py             # Main evaluation script for MedQA benchmark
│
├── data/
│   ├── clean_pubmed.py        # Cleans raw PubMed dataset
│   └── clean_pmc.py           # Cleans raw PMC dataset
│
├── embedding/
│   └── embed_and_index.py     # Embeds documents and builds FAISS indexes
│
├── models/
│   ├── generator_LC.py        # LangChain-based prompt and generation logic
│   ├── load_model.py          # Loads the LLM and tokenizer
│   └── retriever.py           # Document retriever using FAISS
│
├── outputs/                   # Directory for generated outputs (e.g., answers)
└── utils/
    └── metrics.py             # Accuracy, BLEU, F1 evaluation metrics
```

## ⚙️ Pipeline Overview

1. **Data Cleaning**  
   Raw biomedical data from PubMed and PMC are cleaned using `clean_pubmed.py` and `clean_pmc.py`. Output is saved in structured JSON format for further processing.

2. **Chunking and Embedding**  
   Long abstracts are split into overlapping chunks using a sliding window approach. Each chunk is embedded using a domain-specific model like MedCPT. Embeddings are saved and indexed via FAISS.

3. **Index Creation**  
   FAISS indexes are built separately for PMC and PubMed. Metadata is also stored for context reconstruction.

4. **Retriever**  
   Given a medical question, relevant documents are retrieved using nearest neighbor search from the FAISS index.

5. **Generator**  
   A LangChain-powered prompt template is used to structure the input (query + retrieved context) and generate a diagnosis using an LLM (e.g., MedAlpaca, DeepSeek, etc.).

6. **Evaluation**  
   The `test_scores.py` script runs the full system on the MedQA benchmark and computes Accuracy, F1, and BLEU scores.

## 📌 Key Features

- 📚 Dual Knowledge Bases: Combines PMC (clinical) and PubMed (research) literature.
- 🔍 Semantic Retrieval: Efficient FAISS-based nearest neighbor search.
- 🧾 LangChain Prompting: Structured template improves reasoning and output consistency.
- 🧠 Model Agnostic: Compatible with any HuggingFace-supported causal LLM.
- ✅ Evaluation on MedQA: Includes script for real-world medical benchmark testing.

## 🚀 How to Run

1. **Clean the Data**
   ```bash
   python data/clean_pmc.py
   python data/clean_pubmed.py
   ```

2. **Build the FAISS Index**
   ```bash
   python embedding/embed_and_index.py
   ```

3. **Evaluate on MedQA**
   ```bash
   python test_scores.py
   ```

> 💡 Ensure you have the necessary models downloaded and `FAISS_INDEX_*` paths configured in `config.py`.

## 📊 Evaluation Metrics

- **Accuracy**
- **Macro F1-Score**
- **USMLE Pass/Fail Indicator**

## 🔓 License

This project is open-source and released under the MIT License.

---

🧬 Designed for research purposes in biomedical AI. Always validate clinical tools with certified professionals before real-world use.
