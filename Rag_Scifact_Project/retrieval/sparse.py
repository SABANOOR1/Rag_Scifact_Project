"""Traditional keyword baselines: TF-IDF + cosine similarity, and BM25."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

from .base import Retriever


# Hyper-parameters selected on the TRAIN split by notebook 03 (highest nDCG@10), used by the API.
# Notebook 03 asserts that its grid search still re-derives exactly these values.
TFIDF_PARAMS = dict(lowercase=True, stop_words=None, ngram_range=(1, 1), sublinear_tf=True, min_df=1)
BM25_PARAMS = dict(k1=0.9, b=0.5, stop_words="english", ngram_range=(1, 1))


class TfidfRetriever(Retriever):
    """TF-IDF vectors compared with cosine similarity (rows are L2-normalised, so cosine = dot)."""

    name = "tfidf"

    def __init__(self, doc_ids: List[str], texts: List[str], **vectorizer_kwargs):
        params = dict(TFIDF_PARAMS)
        params.update(vectorizer_kwargs)
        self.params = params
        self.doc_ids = list(doc_ids)
        self.vec = TfidfVectorizer(**params)
        self.doc_matrix = self.vec.fit_transform(texts)  # (n_docs, vocab), L2-normalised

    def score(self, queries: Sequence[str]) -> np.ndarray:
        q = self.vec.transform(list(queries))
        return (q @ self.doc_matrix.T).toarray()


class BM25Retriever(Retriever):
    """Okapi BM25 (stronger keyword baseline than raw TF-IDF)."""

    name = "bm25"

    def __init__(self, doc_ids: List[str], texts: List[str], k1: float = BM25_PARAMS["k1"],
                 b: float = BM25_PARAMS["b"], stop_words=BM25_PARAMS["stop_words"],
                 ngram_range=BM25_PARAMS["ngram_range"]):
        self.doc_ids = list(doc_ids)
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(lowercase=True, stop_words=stop_words, ngram_range=ngram_range)
        tf = self.cv.fit_transform(texts).tocsr().astype(np.float32)  # (n_docs, vocab)
        n_docs = tf.shape[0]
        df = np.diff(tf.tocsc().indptr)
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg = doc_len.mean()
        # BM25 weight of each (doc, term): idf * tf*(k1+1) / (tf + k1*(1-b+b*len/avg))
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * doc_len[tf.row] / avg)
        data = self.idf[tf.col] * tf.data * (k1 + 1) / denom
        self.doc_matrix = sp.csr_matrix((data, (tf.row, tf.col)), shape=tf.shape)

    def score(self, queries: Sequence[str]) -> np.ndarray:
        q = self.cv.transform(list(queries))
        q.data[:] = 1.0  # binary query terms
        return (q @ self.doc_matrix.T).toarray()
