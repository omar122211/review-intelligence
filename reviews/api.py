"""FastAPI service for YelpCamp.  Run:  uvicorn reviews.api:app

Local models answer instantly and for free; Gemini is only called for explanations and owner replies.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from reviews.config import ROOT, Settings, get_settings
from reviews.labels import ASPECT_LABELS, ASPECTS
from reviews.llm import CampgroundSummary, LLMError, OwnerReply, ReviewAnalysis, analyze, build_gemini, draft_reply
from reviews.logging_setup import setup_logging
from reviews.service import LocalAnalysis, ReviewIndex, ReviewModels, SearchHit, load_demo

log = logging.getLogger("reviews.api")


class ReviewIn(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("review is too short")
        return v


class ReplyIn(ReviewIn):
    campground: str = Field(default="our campground", max_length=80)


class AnalyzeOut(BaseModel):
    local: LocalAnalysis
    explanation: ReviewAnalysis | None = None
    explanation_error: str | None = None
    latency_ms: int


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def create_app(settings: Settings | None = None, models: ReviewModels | None = None, gemini=None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    state: dict = {"models": models, "gemini": gemini, "campgrounds": {}, "indexes": {}, "insights": {}}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["models"] is None:
            state["models"] = ReviewModels(settings.models_dir, settings.embed_model)
        if state["gemini"] is None:
            try:
                state["gemini"] = build_gemini(settings)
            except LLMError as e:
                log.warning("Gemini features off", extra={"fields": {"reason": str(e)}})
        m: ReviewModels = state["models"]
        insights_file = settings.demo_file.parent / "insights.json"
        state["insights"] = json.loads(insights_file.read_text(encoding="utf-8")) if insights_file.exists() else {}
        for cg in load_demo(settings.demo_file):
            preds = m.sentiment([r["text"] for r in cg["reviews"]]) if cg["reviews"] else []
            for r, p in zip(cg["reviews"], preds):
                r["predicted"] = p.model_dump()
            state["campgrounds"][cg["id"]] = cg
            state["indexes"][cg["id"]] = ReviewIndex(m, cg["reviews"])
        log.info("ready", extra={"fields": {"campgrounds": len(state["campgrounds"]), "llm": state["gemini"] is not None}})
        yield

    app = FastAPI(title="YelpCamp Review Intelligence", version="0.1.0", lifespan=lifespan)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid, started = uuid.uuid4().hex[:12], time.perf_counter()
        response = await call_next(request)
        log.info("request", extra={"fields": {"id": rid, "method": request.method, "path": request.url.path,
                                               "status": response.status_code, "ms": int((time.perf_counter() - started) * 1000)}})
        response.headers["X-Request-ID"] = rid
        return response

    def guard(request: Request, text: str = "") -> None:
        if len(text) > settings.max_review_chars:
            raise HTTPException(422, f"review is longer than {settings.max_review_chars} characters")
        if not limiter.allow(request.client.host if request.client else "unknown"):
            raise HTTPException(429, "Too many requests — please wait a minute.")

    def campground(cg_id: str) -> dict:
        cg = state["campgrounds"].get(cg_id)
        if cg is None:
            raise HTTPException(404, "campground not found")
        return cg

    @app.get("/health")
    def health():
        m = state["models"]
        return {"status": "ok" if m else "starting", "sentiment_model": m.sentiment_model if m else None,
                "llm": state["gemini"] is not None, "campgrounds": len(state["campgrounds"])}

    @app.post("/analyze", response_model=AnalyzeOut)
    def analyze_review(body: ReviewIn, request: Request, explain: bool = False):
        """Sentiment + aspect estimates from the local models; `explain=true` adds Gemini aspect quotes."""
        guard(request, body.text)
        started = time.perf_counter()
        out = {"local": state["models"].analyze(body.text)}
        if explain:
            if state["gemini"] is None:
                out["explanation_error"] = "AI explanations are not configured."
            else:
                try:
                    out["explanation"] = analyze(state["gemini"], body.text)
                except LLMError:
                    log.exception("explain failed")
                    out["explanation_error"] = "AI explanations are unavailable right now."
        return AnalyzeOut(**out, latency_ms=int((time.perf_counter() - started) * 1000))

    @app.post("/reply", response_model=OwnerReply)
    def reply(body: ReplyIn, request: Request):
        """Draft a reply from the campground owner (always reviewed by a human before posting)."""
        guard(request, body.text)
        if state["gemini"] is None:
            raise HTTPException(503, "AI replies are not configured.")
        try:
            return draft_reply(state["gemini"], body.text, body.campground)
        except LLMError:
            log.exception("reply failed")
            raise HTTPException(503, "The AI service is busy right now — please try again shortly.")

    @app.get("/campgrounds")
    def campgrounds():
        out = []
        for cg in state["campgrounds"].values():
            stars = [r["stars"] for r in cg["reviews"]]
            out.append({"id": cg["id"], "name": cg["name"], "location": cg["location"], "reviews": len(stars),
                        "average_stars": round(sum(stars) / len(stars), 2) if stars else None})
        return out

    @app.get("/campgrounds/{cg_id}")
    def campground_insights(cg_id: str):
        cg = campground(cg_id)
        ins = state["insights"].get(cg_id, {})
        aspects = []
        for a in ASPECTS:
            c = ins.get("aspects", {}).get(a, {})
            mentions = sum(c.values())
            aspects.append({"aspect": a, "label": ASPECT_LABELS[a], "mentions": mentions,
                            "positive": c.get("positive", 0), "negative": c.get("negative", 0), "neutral": c.get("neutral", 0),
                            "positive_share": round(c.get("positive", 0) / mentions, 3) if mentions else None})
        summary = ins.get("summary")
        return {"id": cg["id"], "name": cg["name"], "location": cg["location"], "generated_by_ai": cg.get("generated_by_ai", False),
                "summary": CampgroundSummary.model_validate(summary).model_dump() if summary else None,
                "aspects": aspects, "reviews": cg["reviews"],
                "explanations": ins.get("reviews", {})}

    @app.get("/campgrounds/{cg_id}/search", response_model=list[SearchHit])
    def search(cg_id: str, request: Request, q: str = Query(min_length=2, max_length=200), k: int = Query(5, ge=1, le=10)):
        """Find reviews by meaning, e.g. "is it quiet at night?" finds "we slept like babies"."""
        guard(request)
        campground(cg_id)
        return state["indexes"][cg_id].search(q, k)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unexpected error")
        return JSONResponse(status_code=500, content={"detail": "Something went wrong."})

    return app


app = create_app()
