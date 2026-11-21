from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def top_k_from_scores(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """scores: (n_queries, n_docs) dense array -> (indices, scores) of the top-k per row, best first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.take_along_axis(scores, idx, axis=1)


class Retriever:
    """Common interface. Subclasses implement `score(queries) -> (n_queries, n_docs)`."""

    name: str = "retriever"
    doc_ids: List[str]

    def score(self, queries: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def search(self, queries: Sequence[str], k: int = 10):
        """-> list (per query) of [(doc_id, score), ...] best first."""
        idx, sc = top_k_from_scores(self.score(queries), k)
        return [
            [(self.doc_ids[j], float(s)) for j, s in zip(row_i, row_s)]
            for row_i, row_s in zip(idx, sc)
        ]

    def run(self, qid_to_query: dict, k: int = 100) -> dict:
        """Batch search -> {qid: [doc_id, ...]} ready for evaluate_run."""
        qids = list(qid_to_query)
        res = self.search([qid_to_query[q] for q in qids], k)
        return {q: [d for d, _ in r] for q, r in zip(qids, res)}
