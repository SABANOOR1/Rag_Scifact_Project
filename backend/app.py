# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from backend.retriever import retrieve_documents
from backend.generator import generate_answer

app = FastAPI(title="Scientific Document RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    method: Optional[str] = "Sentence-BERT (all-MiniLM-L6-v2)"
    top_k: Optional[int] = 3

class DocumentResponse(BaseModel):
    id: str
    title: str
    text: str
    score: float

class RAGResponse(BaseModel):
    query: str
    method: str
    answer: str
    sources: List[DocumentResponse]

@app.get("/health")
def health_check():
    return {"status": "operational", "system": "SciDoc-QA"}

@app.post("/api/ask", response_model=RAGResponse)
def ask_question(request: QueryRequest):
    # 1. Retrieval phase (using real SciFact corpus & embeddings)
    docs = retrieve_documents(
        query=request.query,
        engine=request.method,
        top_k=request.top_k
    )
    
    # 2. Generation phase
    answer = generate_answer(request.query, docs)
    
    return {
        "query": request.query,
        "method": request.method,
        "answer": answer,
        "sources": docs
    }
