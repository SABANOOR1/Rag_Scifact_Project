# Retrieval layer

Data handling → preprocessing → TF-IDF/BM25 baselines → dense embeddings (MiniLM, BGE) → one `search_scifact()` call for the API.

## Dataset

This project uses the **SciFact** dataset, created by the Allen Institute for AI (AI2) and included in the BEIR benchmark.

- **Source:** https://github.com/allenai/scifact
- **Corpus:** about 5,183 scientific abstracts
- **Claims:** 300 dev claims with SUPPORT / CONTRADICT evidence annotations, used for evaluation
- **License:** claims and evidence annotations are CC BY 4.0; abstracts (S2ORC corpus) are ODC-By 1.0
- **Citation:** Wadden et al., 2020, "Fact or Fiction: Verifying Scientific Claims"

### Getting the data

Download the dataset from the link above and place the files in the `data/` folder before running the notebooks.

## Setup

```bash
pip install -r requirements.txt -r requirements-retrieval.txt
python scripts/build_embeddings.py        # once; downloads MiniLM + BGE, writes models/*.npy  (~8 MB each)
python tests/test_retrieval.py            # 7 tests; no model download needed
```

TF-IDF and BM25 work immediately (no embeddings needed).

## Layout

| Path                                            | Purpose                                                                   |
| ----------------------------------------------- | ------------------------------------------------------------------------- |
| `retrieval/data.py`                             | load corpus/claims, build relevance judgements (qrels)                    |
| `retrieval/sparse.py`                           | `TfidfRetriever`, `BM25Retriever` (+ tuned `TFIDF_PARAMS`, `BM25_PARAMS`) |
| `retrieval/dense.py`                            | model registry `DENSE_MODELS`, `DenseRetriever`, embedding cache          |
| `retrieval/evaluate.py`                         | Recall@k, Hit@k, MRR@10, nDCG@10, MAP; `save_run` / `save_metrics`        |
| `retrieval/search.py`                           | `search_scifact(query, method, top_k)` → `[{id,title,text,score}]`        |
| `notebooks/01–05`                               | exploration, preprocessing, TF-IDF/BM25, MiniLM, BGE                      |
| `results/runs/*.json`, `results/metrics/*.json` | outputs for notebook 06 (Saba)                                            |

## Methods accepted by the API (case-insensitive)

`tfidf` / `TF-IDF Baseline`, `bm25` / `BM25`, `minilm` / `Sentence-BERT (all-MiniLM-L6-v2)`, `bge` / `BGE (bge-small-en-v1.5)`. Unknown names raise `ValueError`; missing embeddings raise `FileNotFoundError` (Najaf can map these to HTTP 400 / 503).
Note: the GUI's current "multi-qa-MiniLM" option is not a model in this package (the notebooks use MiniLM and BGE).

## Protocol

Relevant doc = doc with annotated evidence (SUPPORT/CONTRADICT). Tune on **train** (505 scorable claims), report on **dev** (188). Test claims are unlabeled.

## Results so far (dev, 188 claims)

| Model                    | Recall@10               | MRR@10 | nDCG@10 |
| ------------------------ | ----------------------- | ------ | ------- |
| TF-IDF (sklearn default) | 0.844                   | 0.677  | 0.712   |
| TF-IDF (tuned on train)  | 0.910                   | 0.736  | 0.772   |
| BM25 (tuned on train)    | 0.924                   | 0.785  | 0.815   |
| MiniLM / BGE             | _run notebooks 04 / 05_ |        |         |

## Hand-off to Najaf (backend/GUI owner) — not applied by me

`backend/mock_retriever.py` says to swap in `search_scifact`. The return shape already matches `DocumentResponse`:

```python
# backend/app.py
from retrieval import search_scifact
...
docs = search_scifact(request.query, method=request.method, top_k=request.top_k)
```

and the Streamlit dropdown options should be `Sentence-BERT (all-MiniLM-L6-v2)`, `BGE (bge-small-en-v1.5)`, `BM25`, `TF-IDF Baseline`.
