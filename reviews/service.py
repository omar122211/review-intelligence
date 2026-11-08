"""Local inference: sentiment (fine-tuned MiniLM), aspect estimates (embedding heads) and semantic search."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from pydantic import BaseModel

from reviews.labels import ASPECT_LABELS, ASPECTS, SENTIMENTS


class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class AspectEstimate(BaseModel):
    aspect: str
    label: str
    sentiment: str
    confidence: float


class LocalAnalysis(BaseModel):
    sentiment: SentimentResult
    aspects: list[AspectEstimate]
    model: str


class SearchHit(BaseModel):
    review_id: int
    score: float
    stars: int
    author: str
    text: str


class ReviewModels:
    """Everything that runs on the server without an AI API: fast, free and private."""

    def __init__(self, models_dir: Path, embed_model: str):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        bundle = joblib.load(models_dir / "heads.joblib")
        self.heads = bundle["heads"]
        self.aspect_models = joblib.load(models_dir / "classic.joblib")["aspects"]  # TF-IDF: best on aspects
        self.embedder = SentenceTransformer(embed_model, device="cpu")
        ft = models_dir / "sentiment-minilm"
        if ft.exists():
            self.tok = AutoTokenizer.from_pretrained(ft)
            self.clf = AutoModelForSequenceClassification.from_pretrained(ft, dtype=torch.float32).eval()
            self.sentiment_model = "fine-tuned MiniLM"
        else:  # fall back to the embedding head if the fine-tuned model isn't there
            self.tok = self.clf = None
            self.sentiment_model = "MiniLM embeddings + logistic regression"

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    @torch.inference_mode()
    def sentiment(self, texts: list[str], vecs: np.ndarray | None = None) -> list[SentimentResult]:
        if self.clf is not None:
            enc = self.tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
            probs = self.clf(**enc).logits.float().softmax(-1).numpy()
            classes = [self.clf.config.id2label[i] for i in range(probs.shape[1])]
        else:
            vecs = self.embed(texts) if vecs is None else vecs
            probs, classes = self.heads["sentiment"].predict_proba(vecs), list(self.heads["sentiment"].classes_)
        out = []
        for row in probs:
            p = {c: round(float(v), 4) for c, v in zip(classes, row)}
            best = max(p, key=p.get)
            out.append(SentimentResult(sentiment=best, confidence=p[best], probabilities=p))
        return out

    def analyze(self, text: str) -> LocalAnalysis:
        vec = self.embed([text])
        aspects = []
        for aspect in ASPECTS:
            head = self.aspect_models[aspect]
            probs = head.predict_proba([text])[0]
            i = int(probs.argmax())
            aspects.append(AspectEstimate(aspect=aspect, label=ASPECT_LABELS[aspect], sentiment=str(head.classes_[i]),
                                          confidence=round(float(probs[i]), 4)))
        return LocalAnalysis(sentiment=self.sentiment([text], vec)[0], aspects=aspects, model=self.sentiment_model)


class ReviewIndex:
    """Semantic search over one campground's reviews: finds reviews by meaning, not by exact words."""

    def __init__(self, models: ReviewModels, reviews: list[dict]):
        self.reviews = reviews
        self.vecs = models.embed([r["text"] for r in reviews]) if reviews else np.zeros((0, 384))
        self.models = models

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if not self.reviews:
            return []
        q = self.models.embed([query])[0]
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [SearchHit(review_id=self.reviews[i]["id"], score=round(float(scores[i]), 3), stars=self.reviews[i]["stars"],
                          author=self.reviews[i]["author"], text=self.reviews[i]["text"]) for i in top]


def load_demo(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


__all__ = ["ReviewModels", "ReviewIndex", "load_demo", "SENTIMENTS"]
