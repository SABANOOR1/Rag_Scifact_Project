"""Dataset handling for SciFact (corpus, claims, relevance judgements)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

_WS = re.compile(r"\s+")


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(s: str) -> str:
    """Light normalisation that is safe for ALL models: collapse whitespace."""
    return _WS.sub(" ", s).strip()


def _join_title(title: str, abstract: str) -> str:
    """'title. abstract' without doubling punctuation (most SciFact titles already end in '.')."""
    if not title:
        return abstract
    sep = " " if title[-1] in ".?!:" else ". "
    return f"{title}{sep}{abstract}"


def load_corpus(data_dir: Path = DATA_DIR) -> List[dict]:
    """Load the 5,183 SciFact abstracts.

    Each returned dict has:
        id        str   (doc ids are strings everywhere in this project, matching the API)
        title     str
        abstract  str   (sentences joined by a space)
        sentences list  (original abstract sentences, kept for evidence analysis)
        text      str   "title. abstract"  <- what every retriever indexes
    """
    docs = []
    for d in _read_jsonl(Path(data_dir) / "corpus.jsonl"):
        title = clean_text(d["title"])
        sents = [clean_text(s) for s in d["abstract"] if s.strip()]
        abstract = " ".join(sents)
        docs.append(
            {
                "id": str(d["doc_id"]),
                "title": title,
                "abstract": abstract,
                "sentences": sents,
                "text": _join_title(title, abstract),
            }
        )
    return docs


def load_claims(split: str, data_dir: Path = DATA_DIR) -> List[dict]:
    """split in {'train','dev','test'}. Test claims carry no labels (standard SciFact)."""
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be 'train', 'dev' or 'test'")
    claims = _read_jsonl(Path(data_dir) / f"claims_{split}.jsonl")
    for c in claims:
        c["claim"] = clean_text(c["claim"])
    return claims


def build_qrels(claims: List[dict]) -> Dict[str, Set[str]]:
    """Relevant documents per claim = docs that contain annotated evidence
    (SUPPORT or CONTRADICT). Claims with no evidence (NOT_ENOUGH_INFO) are excluded:
    they have no relevant document, so they cannot be scored for retrieval.

    NOTE: `cited_doc_ids` is NOT used. For NEI claims it lists the document the claim
    was written from, which is not evidence for it.
    """
    qrels: Dict[str, Set[str]] = {}
    for c in claims:
        rel = {str(doc_id) for doc_id in c.get("evidence", {})}
        if rel:
            qrels[str(c["id"])] = rel
    return qrels
