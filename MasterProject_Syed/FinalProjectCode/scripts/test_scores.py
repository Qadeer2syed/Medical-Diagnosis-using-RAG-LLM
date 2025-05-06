import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import sys
import os
import torch
import numpy as np
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
from tqdm import tqdm
#NOTE: Comment the line below to test the model outside the RAG framework
from models.generator_LC import generate_answer
#NOTE: Uncomment the line below to test the model outside the RAG framework
#from models.generateNR import generate_answer
import re


def calculate_usmle_pass_rate(accuracy):
    # USMLE Step 1 passing threshold (2023 standard)
    return accuracy >= 0.60  # 60% minimum passing score

def calculate_ddf1(true, preds):
    # Diagnostic Dialogue F1 (medical specific metric)
    return f1_score(true, preds, average='macro')  # Disease-aware weighting

SAMPLE_FRACTION = 0.01

#Extract the correct answer from generated output
def extract_final_answer(output_text):
    """
    Extracts the predicted answer from the model's output.
    Assumes the format is '[Option Letter] - Explanation'.
    """
    #Extract the correct answer from the output generated
    match = re.match(r"^\s*([A-E])(?:\s*-\s*.*)?$", output_text.strip(), re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # If the model doesn't output a valid option, return an empty string.
        return ''


def evaluate_medqa_hf():
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)["test"]
    dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * SAMPLE_FRACTION)))
    true_labels = []
    pred_labels = []
    bleu_scores = []
    

    for item in tqdm(dataset, desc="Evaluating"):
        # Generate answer
        prompt = (
            f"Medical Question: {item['question']}\n"
            "Options:\n" + "\n".join(
                f"{chr(65+i)}) {option}" for i, option in enumerate(item['options'])
            ) + "\nAnswer:"
        )

        raw_output = generate_answer(prompt)
        
        print(f"[Raw Output]:\n{raw_output}\n")

        # Extract prediction using the extract_final_answer function
        pred = extract_final_answer(raw_output)
        
        if pred is None:  # If the model doesn't provide a valid answer
            pred = ''

        true_answer_text = item['answer']
        
        #Extract the final option letter
        true_answer_key = ''
        for option in item['options']:
            if option['value'].strip().lower() == true_answer_text.strip().lower():
                true_answer_key = option['key']  # Get the letter (A, B, C, D, or E)
                break

        if true_answer_key == '':
            print(f"Warning: Could not find the correct label for '{true_answer_text}'")
        
        # Add the true label to the list
        true_labels.append(true_answer_key)
        
        pred_labels.append(pred)
    
    print(true_labels)
    print(pred_labels)
    
    # Convert labels to numerical form for F1 calculation
    label_map = {'A':0, 'B':1, 'C':2, 'D':3}
    y_true = [label_map.get(t, -1) for t in true_labels]
    y_pred = [label_map.get(p, -1) for p in pred_labels]

    # Filter out invalid predictions
    valid_idx = [i for i, p in enumerate(y_pred) if p != -1]
    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]

    # Calculate metrics
    accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)])
    f1 = f1_score(y_true, y_pred, average='macro')
    usmle_pass = calculate_usmle_pass_rate(accuracy)
    
    # Display results
    print(f"\nMedical QA Evaluation Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1-Score (Macro): {f1:.3f}")
    print(f"USMLE Pass Rate: {'PASS' if usmle_pass else 'FAIL'}")

if __name__ == "__main__":
    evaluate_medqa_hf()

