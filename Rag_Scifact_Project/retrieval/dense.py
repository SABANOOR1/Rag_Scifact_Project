"""Dense (embedding) retrieval: encode abstracts once, cache to disk, rank by cosine similarity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import Retriever

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# key -> config.  Add a model here and it is available everywhere (scripts, notebooks, API).
DENSE_MODELS = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "max_seq_length": 256,   # model's native limit
    },
    "bge": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "max_seq_length": 512,
    },
}


def load_sentence_transformer(key: str):
    """Load the real SentenceTransformer for a registry key (needs `pip install sentence-transformers`)."""
    from sentence_transformers import SentenceTransformer  # lazy: keeps TF-IDF usable without torch

    cfg = DENSE_MODELS[key]
    model = SentenceTransformer(cfg["hf_name"])
    model.max_seq_length = cfg["max_seq_length"]
    return model


def encode_corpus(encoder, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
    """Encode documents -> float32, L2-normalised (so dot product == cosine similarity)."""
    emb = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


class DenseRetriever(Retriever):
    def __init__(self, name: str, doc_ids: List[str], doc_embeddings: np.ndarray, encoder,
                 query_prefix: str = "", use_faiss: bool = False):
        assert len(doc_ids) == doc_embeddings.shape[0], "ids / embeddings length mismatch"
        self.name = name
        self.doc_ids = list(doc_ids)
        self.doc_embeddings = doc_embeddings
        self.encoder = encoder
        self.query_prefix = query_prefix
        self._faiss_index = None
        if use_faiss:  # optional; exact inner-product index, same ranking as the numpy path
            import faiss

            self._faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
            self._faiss_index.add(doc_embeddings)

    def _encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        qs = [self.query_prefix + q for q in queries]
        return encode_corpus(self.encoder, qs, show_progress_bar=False)

    def score(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode_queries(queries) @ self.doc_embeddings.T

    def search(self, queries: Sequence[str], k: int = 10):
        if self._faiss_index is None:
            return super().search(queries, k)
        sc, idx = self._faiss_index.search(self._encode_queries(queries), min(k, len(self.doc_ids)))
        return [[(self.doc_ids[j], float(s)) for j, s in zip(ri, rs)] for ri, rs in zip(idx, sc)]

    # ---- persistence -------------------------------------------------------------------
    @staticmethod
    def paths(key: str, models_dir: Path = MODELS_DIR):
        return models_dir / f"{key}_embeddings.npy", models_dir / f"{key}_doc_ids.json"

    @staticmethod
    def save_embeddings(key: str, doc_ids: List[str], emb: np.ndarray, models_dir: Path = MODELS_DIR):
        models_dir.mkdir(parents=True, exist_ok=True)
        e, i = DenseRetriever.paths(key, models_dir)
        np.save(e, emb)
        with open(i, "w") as f:
            json.dump(list(doc_ids), f)

    @classmethod
    def from_cache(cls, key: str, encoder, models_dir: Path = MODELS_DIR, **kw) -> "DenseRetriever":
        e, i = cls.paths(key, models_dir)
        if not e.exists():
            raise FileNotFoundError(
                f"{e} not found. Run `python scripts/build_embeddings.py` first "
                "(or run notebook 04/05)."
            )
        emb = np.load(e)
        with open(i) as f:
            ids = json.load(f)
        prefix = DENSE_MODELS.get(key, {}).get("query_prefix", "")
        return cls(key, ids, emb, encoder, query_prefix=prefix, **kw)
