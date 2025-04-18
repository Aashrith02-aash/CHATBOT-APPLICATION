import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("squad_faiss.index")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA pairs
from datasets import load_dataset
dataset = load_dataset("squad")
qa_pairs = [(item["question"], item["answers"]["text"][0]) for item in dataset["train"]]

def get_bot_response(user_input):
    # Convert user input to embedding
    user_embedding = model.encode([user_input])
    
    # Search FAISS for the closest question
    _, result_index = index.search(user_embedding, 1)
    
    # Retrieve the best-matching answer
    best_answer = qa_pairs[result_index[0][0]][1]
    
    return best_answer
