"""Single entry point used by the FastAPI backend:  search_scifact(query, method, top_k)."""
from __future__ import annotations

import threading
from typing import Dict, List

from .data import load_corpus
from .dense import DENSE_MODELS, DenseRetriever, load_sentence_transformer
from .sparse import BM25Retriever, TfidfRetriever

# What the GUI / API may send  ->  canonical key. Matching is case-insensitive.
METHOD_ALIASES = {
    "tf-idf baseline": "tfidf",
    "tf-idf": "tfidf",
    "tfidf": "tfidf",
    "bm25": "bm25",
    "sentence-bert (all-minilm-l6-v2)": "minilm",
    "sentence-bert": "minilm",   # backend's current default value
    "minilm": "minilm",
    "all-minilm-l6-v2": "minilm",
    "bge": "bge",
    "bge (bge-small-en-v1.5)": "bge",
    "bge-small": "bge",
    "bge-small-en-v1.5": "bge",
}
AVAILABLE_METHODS = ["minilm", "bge", "tfidf", "bm25"]

_lock = threading.Lock()
_corpus: List[dict] | None = None
_by_id: Dict[str, dict] = {}
_retrievers: Dict[str, object] = {}


def resolve_method(method: str) -> str:
    key = METHOD_ALIASES.get((method or "").strip().lower())
    if key is None:
        raise ValueError(f"Unknown retrieval method '{method}'. Valid: {sorted(set(METHOD_ALIASES))}")
    return key


def _load_corpus_once():
    global _corpus, _by_id
    if _corpus is None:
        _corpus = load_corpus()
        _by_id = {d["id"]: d for d in _corpus}
    return _corpus


def get_retriever(key: str):
    """Lazily build / load each retriever once (thread-safe) and keep it in memory."""
    with _lock:
        if key in _retrievers:
            return _retrievers[key]
        corpus = _load_corpus_once()
        ids = [d["id"] for d in corpus]
        texts = [d["text"] for d in corpus]
        if key == "tfidf":
            r = TfidfRetriever(ids, texts)
        elif key == "bm25":
            r = BM25Retriever(ids, texts)
        elif key in DENSE_MODELS:
            if not DenseRetriever.paths(key)[0].exists():   # fail fast, before loading a model
                raise FileNotFoundError(
                    f"No cached embeddings for '{key}'. Run `python scripts/build_embeddings.py {key}` first."
                )
            r = DenseRetriever.from_cache(key, load_sentence_transformer(key))
        else:
            raise ValueError(key)
        _retrievers[key] = r
        return r


def search_scifact(query: str, method: str = "minilm", top_k: int = 3) -> List[dict]:
    """Return the top_k most relevant abstracts as
    [{"id": str, "title": str, "text": str, "score": float}, ...]  (best first)
    which is exactly the shape backend.app.DocumentResponse expects.
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    key = resolve_method(method)
    retriever = get_retriever(key)
    hits = retriever.search([query.strip()], k=top_k)[0]
    return [
        {
            "id": doc_id,
            "title": _by_id[doc_id]["title"],
            "text": _by_id[doc_id]["abstract"],
            "score": score,
        }
        for doc_id, score in hits
    ]
