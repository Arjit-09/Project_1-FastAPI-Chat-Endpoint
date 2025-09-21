from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

openai.api_key = OPENAI_API_KEY

# FastAPI setup
app = FastAPI(title="FAISS + OpenAI Chat API")


# Sample data for FAISS

documents = [
    {"id": 0, "text": "Python is a programming language."},
    {"id": 1, "text": "FastAPI is a modern Python web framework."},
    {"id": 2, "text": "FAISS is a library for efficient similarity search."},
    {"id": 3, "text": "OpenAI API can generate text from prompts."},
]

# Generate embeddings for each document
def get_embedding(text: str):
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

doc_embeddings = np.array([get_embedding(doc["text"]) for doc in documents])
embedding_dim = doc_embeddings.shape[1]

# Initialize FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)

class ChatRequest(BaseModel):
    query: str
    k: int = 2  # Number of FAISS results to retrieve

class ChatResponse(BaseModel):
    answer: str
    sources: list


# /chat endpoint

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # 1. Embed user query
        query_embedding = get_embedding(request.query).reshape(1, -1)

        # 2. FAISS similarity search
        distances, indices = index.search(query_embedding, request.k)
        retrieved_docs = [documents[i]["text"] for i in indices[0]]

        # 3. Build context prompt
        context_text = "\n".join(retrieved_docs)
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question:\n"
            f"{context_text}\n\nQuestion: {request.query}\nAnswer:"
        )

        # 4. Generate response with OpenAI
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        answer = response.choices[0].message["content"]

        return ChatResponse(answer=answer, sources=retrieved_docs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))