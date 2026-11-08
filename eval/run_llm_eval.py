"""Compare every approach on the same 300 real test reviews (100 negative / neutral / positive), and check
Gemini's aspect opinions against the reviewers' own per-aspect star ratings.

    python -m eval.run_llm_eval       # ~300 API calls, ~25 minutes on the free tier

Writes eval/llm.json and eval/results.md (together with classic.json and finetune.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from reviews.config import ROOT, get_settings
from reviews.labels import ASPECT_LABELS, ASPECTS, stars_to_sentiment
from reviews.llm import LLMError, analyze, build_gemini
from reviews.service import ReviewModels

HERE = ROOT / "eval"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--pause", type=float, default=4.0)
    args = ap.parse_args()
    s = get_settings()
    test = pd.read_parquet(s.data_dir / "splits" / "test.parquet")
    sample = pd.concat([test[test["sentiment"] == c].head(args.per_class) for c in ("negative", "neutral", "positive")])
    texts = sample["review"].tolist()

    # local models on the same reviews
    models = ReviewModels(s.models_dir, s.embed_model)
    t0 = time.perf_counter()
    ft = [r.sentiment for r in models.sentiment(texts)]
    ft_ms = (time.perf_counter() - t0) / len(texts) * 1000
    tfidf = joblib.load(s.models_dir / "classic.joblib")["sentiment"].predict(texts)
    emb = models.heads["sentiment"].predict(models.embed(texts))

    # Gemini, one review at a time (free-tier pacing)
    gem = build_gemini(s)
    rows = []
    for i, (text, (_, row)) in enumerate(zip(texts, sample.iterrows()), 1):
        t = time.perf_counter()
        try:
            a = analyze(gem, text)
            pred, aspects = a.sentiment, [m.model_dump() for m in a.aspects]
        except LLMError:
            pred, aspects = None, []
        rows.append({"truth": row["sentiment"], "gemini": pred, "aspects": aspects, "ms": round((time.perf_counter() - t) * 1000),
                     **{f"{a}_stars": float(row[f"{a}_stars"]) for a in ASPECTS}})
        if i % 25 == 0:
            print(f"{i}/{len(texts)}", flush=True)
        time.sleep(args.pause)

    truth = sample["sentiment"].tolist()
    ok = [i for i, r in enumerate(rows) if r["gemini"] is not None]
    comp = {
        "TF-IDF + logistic regression": {"pred": list(tfidf)},
        "MiniLM embeddings + logistic regression": {"pred": list(emb)},
        "Fine-tuned MiniLM": {"pred": ft, "ms": round(ft_ms, 1)},
        "Gemini zero-shot": {"pred": [r["gemini"] for r in rows], "ms": float(pd.Series([r["ms"] for r in rows]).median())},
    }
    table = {}
    for name, c in comp.items():
        idx = ok if name == "Gemini zero-shot" else range(len(truth))
        y, p = [truth[i] for i in idx], [c["pred"][i] for i in idx]
        table[name] = {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4),
                       "n": len(y), "ms": c.get("ms")}

    # aspect check: when Gemini says an aspect is positive/negative, does the reviewer's own aspect star agree?
    aspect_rows = {}
    for a in ASPECTS:
        clear = [(m["sentiment"], stars_to_sentiment(r[f"{a}_stars"])) for r in rows for m in r["aspects"]
                 if m["aspect"] == a and m["sentiment"] != "neutral" and r[f"{a}_stars"] != 3]
        aspect_rows[a] = {"mentions": sum(m["aspect"] == a for r in rows for m in r["aspects"]),
                          "checked": len(clear), "agree": sum(p == t for p, t in clear)}
    result = {"sentiment": table, "aspects": aspect_rows, "api_errors": len(rows) - len(ok), "n": len(rows)}
    (HERE / "llm.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_report(result)
    print(json.dumps(table, indent=1))


def write_report(llm: dict) -> None:
    classic = json.loads((HERE / "classic.json").read_text(encoding="utf-8"))
    ft = json.loads((HERE / "finetune.json").read_text(encoding="utf-8"))
    lines = ["# Results\n", "All numbers are on **real reviews never seen in training** (TripAdvisor, reviewers' own star ratings as labels; "
             "classes balanced, so always guessing \"positive\" would score 33 %).\n",
             "## Overall sentiment — full test set (3,000 reviews)\n",
             "| Approach | Accuracy | Macro F1 |", "|---|---|---|",
             f"| TF-IDF + logistic regression | {classic['tfidf_lr']['sentiment']['accuracy']:.1%} | {classic['tfidf_lr']['sentiment']['macro_f1']:.3f} |",
             f"| MiniLM embeddings + logistic regression | {classic['embeddings_lr']['sentiment']['accuracy']:.1%} | {classic['embeddings_lr']['sentiment']['macro_f1']:.3f} |",
             f"| **Fine-tuned MiniLM** | **{ft['accuracy']:.1%}** | **{ft['macro_f1']:.3f}** |",
             f"\n## Same 300 test reviews, including Gemini\n",
             "| Approach | Accuracy | Macro F1 | Time per review |", "|---|---|---|---|"]
    for name, r in llm["sentiment"].items():
        ms = f"{r['ms']:.0f} ms" if r.get("ms") else "–"
        lines.append(f"| {name} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | {ms} |")
    lines += ["\n## Aspects\n",
              "Gemini lists only the aspects a review talks about, with a quote. When it calls an aspect positive or negative, "
              "does the reviewer's own star rating for that aspect agree? (Only clear cases: aspect stars ≠ 3.)\n",
              "| Aspect | Mentioned in | Agreement with reviewer's aspect stars |", "|---|---|---|"]
    for a, r in llm["aspects"].items():
        agree = f"{r['agree']}/{r['checked']} ({r['agree'] / r['checked']:.0%})" if r["checked"] else "–"
        lines.append(f"| {ASPECT_LABELS[a]} | {r['mentions']}/{llm['n']} reviews | {agree} |")
    lines += ["\nThe local aspect heads (MiniLM embeddings + logistic regression) predict every aspect for every review, "
              "mentioned or not — macro F1 on the full test set:\n", "| Aspect | Embeddings | TF-IDF |", "|---|---|---|"]
    for a in ASPECTS:
        lines.append(f"| {ASPECT_LABELS[a]} | {classic['embeddings_lr'][a]['macro_f1']:.3f} | {classic['tfidf_lr_aspects'][a]['macro_f1']:.3f} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
