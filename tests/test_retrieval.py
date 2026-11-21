"""Run with:  python tests/test_retrieval.py     (also works under pytest)

The dense-path tests use a small LSA 'encoder' that mimics SentenceTransformer.encode(),
so they run with no torch / no model download. They check the plumbing (caching, ranking,
prefixes, API shape), not the quality of MiniLM/BGE.
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.data import build_qrels, load_claims, load_corpus
from retrieval.dense import DenseRetriever, encode_corpus
from retrieval.evaluate import evaluate_run
from retrieval.search import resolve_method, search_scifact
from retrieval.sparse import BM25Retriever, TfidfRetriever

CORPUS = load_corpus()
IDS = [d["id"] for d in CORPUS]
TEXTS = [d["text"] for d in CORPUS]


class LSAEncoder:
    """Stand-in for SentenceTransformer: same .encode() signature."""

    def __init__(self, dim=64):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.v = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.svd = TruncatedSVD(dim, random_state=0)
        self.svd.fit(self.v.fit_transform(TEXTS))
        self.seen = []

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True):
        self.seen.extend(texts[:1])
        e = self.svd.transform(self.v.transform(texts))
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.clip(n, 1e-9, None)


def test_data_sizes():
    assert len(CORPUS) == 5183 and len(set(IDS)) == 5183
    assert len(build_qrels(load_claims("train"))) == 505
    assert len(build_qrels(load_claims("dev"))) == 188
    assert build_qrels(load_claims("test")) == {}          # test is unlabeled
    assert all(isinstance(i, str) for i in IDS)


def test_metrics_hand_computed():
    m = evaluate_run({"a": ["z", "x", "w", "y"]}, {"a": {"x", "y"}}, ks=(1, 2, 4))
    ideal = 1 + 1 / math.log2(3)
    assert abs(m["nDCG@10"] - (1 / math.log2(3) + 1 / math.log2(5)) / ideal) < 1e-9
    assert m["MRR@10"] == 0.5 and m["Recall@1"] == 0 and m["Recall@2"] == 0.5 and m["Recall@4"] == 1
    assert abs(m["MAP"] - 0.5) < 1e-9


def test_sparse_self_retrieval():
    for R in (TfidfRetriever, BM25Retriever):
        r = R(IDS, TEXTS)
        res = r.search([CORPUS[10]["title"]], 5)[0]
        assert res[0][0] == CORPUS[10]["id"], R.__name__
        assert [s for _, s in res] == sorted([s for _, s in res], reverse=True)


def test_dense_cache_roundtrip_and_search():
    enc = LSAEncoder()
    emb = encode_corpus(enc, TEXTS, show_progress_bar=False)
    assert emb.dtype == np.float32 and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    with tempfile.TemporaryDirectory() as d:
        DenseRetriever.save_embeddings("stub", IDS, emb, Path(d))
        r = DenseRetriever.from_cache("stub", enc, Path(d))
    assert r.doc_ids == IDS
    # querying with a document's own text must return that document with cosine ~= 1
    top = r.search([TEXTS[42]], 5)[0]
    assert top[0][0] == CORPUS[42]["id"] and top[0][1] > 0.999
    assert [s for _, s in top] == sorted([s for _, s in top], reverse=True)
    full = r.score([TEXTS[42]])[0]
    assert abs(top[0][1] - full.max()) < 1e-6


def test_query_prefix_applied_only_to_queries():
    enc = LSAEncoder()
    emb = encode_corpus(enc, TEXTS[:50], show_progress_bar=False)
    r = DenseRetriever("x", IDS[:50], emb, enc, query_prefix="PFX: ")
    enc.seen.clear()
    r.search(["hello world"], 3)
    assert enc.seen == ["PFX: hello world"]


def test_api_shape_and_aliases():
    for alias, key in [("TF-IDF Baseline", "tfidf"), ("Sentence-BERT", "minilm"),
                       ("Sentence-BERT (all-MiniLM-L6-v2)", "minilm"), ("BGE", "bge"), ("bm25", "bm25")]:
        assert resolve_method(alias) == key
    try:
        resolve_method("multi-qa-MiniLM")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    out = search_scifact("Do statins reduce the risk of stroke?", "TF-IDF Baseline", top_k=3)
    assert len(out) == 3
    assert set(out[0]) == {"id", "title", "text", "score"}
    assert isinstance(out[0]["id"], str) and isinstance(out[0]["score"], float)


def test_empty_query_rejected():
    try:
        search_scifact("   ", "tfidf")
        raise AssertionError
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")
