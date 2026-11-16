"""Encode the SciFact corpus with each dense model and cache embeddings to models/.

    python scripts/build_embeddings.py                 # all models
    python scripts/build_embeddings.py minilm          # one model
    python scripts/build_embeddings.py bge --batch 16  # smaller batch if you run out of RAM

~5k abstracts: a few minutes on CPU, <1 min on a Colab GPU.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.data import load_corpus
from retrieval.dense import DENSE_MODELS, DenseRetriever, encode_corpus, load_sentence_transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(DENSE_MODELS))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    corpus = load_corpus()
    ids = [d["id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    for key in args.models:
        t0 = time.time()
        print(f"[{key}] loading {DENSE_MODELS[key]['hf_name']} ...")
        enc = load_sentence_transformer(key)
        emb = encode_corpus(enc, texts, batch_size=args.batch)
        DenseRetriever.save_embeddings(key, ids, emb)
        print(f"[{key}] saved {emb.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
