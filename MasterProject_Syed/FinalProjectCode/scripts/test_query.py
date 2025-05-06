# scripts/test_query.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
#NOTE: Comment the line below to test the model outside the RAG framework
from models.generator_LC import generate_answer
#NOTE: Uncomment the line below to test the model outside the RAG framework
#from models.generateNR import generate_answer

if __name__ == "__main__":
    query = input("Enter your medical query: ")
    answer = generate_answer(query)
    print("\nGenerated Answer:\n", answer)

