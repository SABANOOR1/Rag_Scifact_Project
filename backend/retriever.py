# backend/retriever.py
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.jsonl")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 1. Load the SciFact corpus into memory
corpus_by_id = {}
all_doc_ids = []
all_texts = []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        doc_id = str(item["doc_id"])
        title = item.get("title", "")
        abstract_text = " ".join(item.get("abstract", []))
        corpus_by_id[doc_id] = {
            "id": doc_id,
            "title": title,
            "text": abstract_text,
        }
        all_doc_ids.append(doc_id)
        all_texts.append(f"{title} {abstract_text}")

# 2. Load precomputed MiniLM & BGE embeddings saved by teammates
minilm_embeddings = np.load(os.path.join(MODELS_DIR, "minilm_embeddings.npy"))
with open(os.path.join(MODELS_DIR, "minilm_doc_ids.json"), "r") as f:
    minilm_doc_ids = [str(d) for d in json.load(f)]

bge_embeddings = np.load(os.path.join(MODELS_DIR, "bge_embeddings.npy"))
with open(os.path.join(MODELS_DIR, "bge_doc_ids.json"), "r") as f:
    bge_doc_ids = [str(d) for d in json.load(f)]

# 3. Build TF-IDF index over the corpus
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

# Lazy-load transformer encoders so startup stays fast
_encoders = {}

def _get_encoder(model_name: str) -> SentenceTransformer:
    if model_name not in _encoders:
        _encoders[model_name] = SentenceTransformer(model_name)
    return _encoders[model_name]


def retrieve_documents(query: str, engine: str = "Sentence-BERT (all-MiniLM-L6-v2)", top_k: int = 3) -> list:
    """
    Retrieves the top_k most relevant SciFact documents using the selected engine.
    """
    engine_lower = engine.lower()

    if "tfidf" in engine_lower or "tf-idf" in engine_lower:
        q_vec = tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, tfidf_matrix)[0]
        doc_ids = all_doc_ids

    elif "bge" in engine_lower:
        encoder = _get_encoder("BAAI/bge-small-en-v1.5")
        q_vec = encoder.encode([query], normalize_embeddings=True)
        scores = np.dot(bge_embeddings, q_vec.T).squeeze()
        doc_ids = bge_doc_ids

    else:
        # Default: Sentence-BERT (all-MiniLM-L6-v2)
        encoder = _get_encoder("all-MiniLM-L6-v2")
        q_vec = encoder.encode([query], normalize_embeddings=True)
        scores = np.dot(minilm_embeddings, q_vec.T).squeeze()
        doc_ids = minilm_doc_ids

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc_id = doc_ids[idx]
        doc = corpus_by_id.get(doc_id)
        if doc:
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": round(float(scores[idx]), 4),
            })

    return results