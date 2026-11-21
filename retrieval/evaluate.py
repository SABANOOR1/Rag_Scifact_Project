"""Ranking metrics for retrieval (binary relevance, per-query then averaged)."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set

import numpy as np


def _dcg(rels: Sequence[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    ks: Sequence[int] = (1, 5, 10, 20, 50, 100),
    ndcg_k: int = 10,
    mrr_k: int = 10,
) -> Dict[str, float]:
    """run:   qid -> ranked list of doc ids (best first)
    qrels: qid -> set of relevant doc ids
    Only queries present in qrels are scored.
    """
    recalls = {k: [] for k in ks}
    hits = {k: [] for k in ks}
    rr, ndcg, ap = [], [], []
    for qid, rel in qrels.items():
        ranked = run.get(qid, [])
        flags = [1 if d in rel else 0 for d in ranked]
        for k in ks:
            found = sum(flags[:k])
            recalls[k].append(found / len(rel))
            hits[k].append(1.0 if found else 0.0)
        # MRR@k
        first = next((i for i, f in enumerate(flags[:mrr_k]) if f), None)
        rr.append(1.0 / (first + 1) if first is not None else 0.0)
        # nDCG@k
        ideal = _dcg([1] * min(len(rel), ndcg_k))
        ndcg.append(_dcg(flags[:ndcg_k]) / ideal if ideal else 0.0)
        # AP (over the whole supplied ranking)
        seen, prec_sum = 0, 0.0
        for i, f in enumerate(flags):
            if f:
                seen += 1
                prec_sum += seen / (i + 1)
        ap.append(prec_sum / len(rel))
    out = {"n_queries": len(qrels)}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recalls[k]))
        out[f"Hit@{k}"] = float(np.mean(hits[k]))
    out[f"MRR@{mrr_k}"] = float(np.mean(rr))
    out[f"nDCG@{ndcg_k}"] = float(np.mean(ndcg))
    out["MAP"] = float(np.mean(ap))
    return out


def per_query_first_hit_rank(run, qrels) -> Dict[str, int | None]:
    """Rank (1-based) of the first relevant doc per query, None if absent.
    Useful for error analysis."""
    res = {}
    for qid, rel in qrels.items():
        res[qid] = next((i + 1 for i, d in enumerate(run.get(qid, [])) if d in rel), None)
    return res


# ---- persistence of results (shared contract with the evaluation notebook) -------------
import json as _json
from pathlib import Path as _Path

RESULTS_DIR = _Path(__file__).resolve().parent.parent / "results"


def save_run(name: str, split: str, run: Dict[str, List[str]]) -> _Path:
    """results/runs/<name>_<split>.json : {qid: [top-100 doc ids]}"""
    p = RESULTS_DIR / "runs" / f"{name}_{split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(run))
    return p


def load_run(name: str, split: str) -> Dict[str, List[str]]:
    return _json.loads((RESULTS_DIR / "runs" / f"{name}_{split}.json").read_text())


def save_metrics(name: str, split: str, metrics: Dict[str, float]) -> _Path:
    """results/metrics/<name>_<split>.json"""
    p = RESULTS_DIR / "metrics" / f"{name}_{split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(metrics, indent=2))
    return p
