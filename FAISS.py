import chromadb

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Store data persistently

# Create a collection
collection = chroma_client.get_or_create_collection(name="squad_qa")

# Add data to ChromaDB
for item in dataset["train"]:
    question = item["question"]
    answer = item["answers"]["text"][0]
    context = item["context"]

    # Generate embeddings for the question
    question_embedding = generate_embedding(question)

    # Store in ChromaDB
    collection.add(
        ids=[item["id"]],
        embeddings=[question_embedding],
        metadatas=[{"question": question, "answer": answer, "context": context}]
    )

print("✅ SQuAD data stored in ChromaDB!")
