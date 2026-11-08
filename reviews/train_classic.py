"""Train the fast models: TF-IDF + logistic regression, and sentence embeddings + logistic regression.

    python -m reviews.train_classic      # ~10 minutes on a laptop CPU (most of it is embedding 16k reviews)

Writes models/classic.joblib (TF-IDF sentiment) and models/heads.joblib (embedding heads: sentiment + 5 aspects),
and eval/classic.json.
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from reviews.config import ROOT, get_settings
from reviews.labels import ASPECTS, SENTIMENTS, stars_to_sentiment


def load_splits() -> dict[str, pd.DataFrame]:
    d = get_settings().data_dir / "splits"
    return {k: pd.read_parquet(d / f"{k}.parquet") for k in ("train", "val", "test")}


def embed(texts: list[str], name: str) -> np.ndarray:
    """Sentence embeddings, cached on disk (they take a few minutes on CPU)."""
    s = get_settings()
    cache = s.data_dir / f"emb-{name}.npy"
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(s.embed_model, device="cpu")
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.save(cache, vecs)
    return vecs


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4)}


def best_lr(x_train, y_train, x_val, y_val) -> LogisticRegression:
    """Pick the regularisation strength on the validation set — never on the test set."""
    best = None
    for c in (0.25, 1.0, 4.0, 16.0):
        model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(x_train, y_train)
        f1 = f1_score(y_val, model.predict(x_val), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, model)
    return best[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    s = get_settings()
    data = load_splits()
    results: dict = {}

    # 1) TF-IDF + logistic regression (classic NLP baseline)
    t0 = time.time()
    best = None
    for c in (1.0, 4.0, 16.0):
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150_000, sublinear_tf=True),
                             LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
        pipe.fit(data["train"]["review"], data["train"]["sentiment"])
        f1 = f1_score(data["val"]["sentiment"], pipe.predict(data["val"]["review"]), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, pipe)
    tfidf = best[1]
    results["tfidf_lr"] = {"sentiment": scores(data["test"]["sentiment"], tfidf.predict(data["test"]["review"])),
                           "train_minutes": round((time.time() - t0) / 60, 1)}
    print("tfidf", results["tfidf_lr"], flush=True)

    # 2) Sentence embeddings (MiniLM, frozen) + logistic regression heads
    t0 = time.time()
    x = {k: embed(v["review"].tolist(), k) for k, v in data.items()}
    tfidf_aspects: dict = {}
    heads = {"sentiment": best_lr(x["train"], data["train"]["sentiment"], x["val"], data["val"]["sentiment"])}
    emb_res = {"sentiment": scores(data["test"]["sentiment"], heads["sentiment"].predict(x["test"]))}
    for aspect in ASPECTS:
        y = {k: v[f"{aspect}_stars"].map(stars_to_sentiment) for k, v in data.items()}
        heads[aspect] = best_lr(x["train"], y["train"], x["val"], y["val"])
        emb_res[aspect] = scores(y["test"], heads[aspect].predict(x["test"]))
        # same aspect with TF-IDF for comparison
        a_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150_000, sublinear_tf=True),
                               LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))
        a_pipe.fit(data["train"]["review"], y["train"])
        results.setdefault("tfidf_lr_aspects", {})[aspect] = scores(y["test"], a_pipe.predict(data["test"]["review"]))
        tfidf_aspects[aspect] = a_pipe
    results["embeddings_lr"] = emb_res | {"train_minutes": round((time.time() - t0) / 60, 1)}
    print("embeddings", results["embeddings_lr"], flush=True)
    print("tfidf aspects", results["tfidf_lr_aspects"], flush=True)

    s.models_dir.mkdir(exist_ok=True)
    # TF-IDF beat the frozen embeddings on every aspect, so the service uses these pipelines for aspects
    joblib.dump({"sentiment": tfidf, "aspects": tfidf_aspects, "classes": SENTIMENTS}, s.models_dir / "classic.joblib", compress=3)
    joblib.dump({"embed_model": s.embed_model, "heads": heads, "classes": SENTIMENTS}, s.models_dir / "heads.joblib", compress=3)
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "classic.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
