import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------
# 1. Sample documents
# -------------------------

documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Vector databases store embeddings for similarity search.",
    "Embeddings convert text into numerical vectors.",
    "FAISS is a library for fast vector similarity search.",
]

# -------------------------
# 2. Load embedding model
# -------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# -------------------------
# 3. Create FAISS index
# -------------------------

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# -------------------------
# 4. User question
# -------------------------

question = "What does RAG mean?"

question_embedding = embedding_model.encode([question])
question_embedding = np.array(question_embedding).astype("float32")

# -------------------------
# 5. Retrieve top-k docs
# -------------------------

k = 2
distances, indices = index.search(question_embedding, k)

retrieved_docs = [documents[i] for i in indices[0]]

print("Retrieved docs:")
for doc in retrieved_docs:
    print("-", doc)

# -------------------------
# 6. Build prompt
# -------------------------

context = "\n".join(retrieved_docs)

prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{question}
"""

# -------------------------
# 7. Query OpenRouter
# -------------------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY",
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",   # or anthropic/claude-3.5-sonnet, google/gemini-2.5-pro, etc.
    messages=[
        {"role": "user", "content": prompt}
    ],
)

answer = response.choices[0].message.content

print("\nAnswer:")
print(answer)