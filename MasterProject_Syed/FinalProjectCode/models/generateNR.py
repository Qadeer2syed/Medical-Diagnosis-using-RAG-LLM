import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import torch
from models.load_model import model, tokenizer
from models.retriever import retrieve_documents
from langchain.prompts import PromptTemplate

def generate_answer(query):
    # Retrieve documents: top 3 from each dataset

    # Define the prompt using LangChain's PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=
       """
        You are a highly skilled medical diagnostic AI.
        
        Query: {query}

        Instructions:
        - Only provide the final answer in the format: "Final Answer: [Option Letter] - [Explanation]
        - Provide Differential Diagnosis
        - Provide a concise diagnosis and reasoning.

       RESPONSE:
       """
    )

    prompt = prompt_template.format(query=query)

    # Determine device: use GPU if available, else CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding="max_length"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generate output
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated tokens
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the answer using LangChain's pattern recognition capabilities
    if "Final Answer:" in output_text:
        answer = output_text.split("Final Answer:")[-1].strip()
    else:
        answer = "No valid answer generated."

    return answer

if __name__ == "__main__":
    query = input("Enter your medical query: ")
    print("\nGenerated Answer:\n", generate_answer(query))
