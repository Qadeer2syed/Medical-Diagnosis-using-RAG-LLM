import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import MODEL_NAME
from huggingface_hub import snapshot_download



def load_model():

    #Loading the model and tokenizer using transformers library
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

    # Tokenizer for Mixtral - Uncomment this section for Mixtral implementation
    #if tokenizer.pad_token is None:
       #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
       #model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

# For convenience, you can load these globally:
model, tokenizer = load_model()
