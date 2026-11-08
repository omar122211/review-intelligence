"""Pre-compute Gemini insights for the demo campgrounds, so the dashboard loads instantly and costs nothing.

    python -m demo.precompute     # ~95 API calls, ~8 minutes on the free tier
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

from reviews.config import ROOT, get_settings
from reviews.labels import ASPECTS
from reviews.llm import LLMError, analyze, build_gemini, summarize
from reviews.service import load_demo

OUT = ROOT / "demo" / "insights.json"


def patient(fn, *args, tries: int = 4):
    """Free-tier calls sometimes time out; wait and try again instead of giving up."""
    for attempt in range(tries):
        try:
            return fn(*args)
        except LLMError:
            if attempt == tries - 1:
                raise
            time.sleep(30 * (attempt + 1))


def main(pause: float = 4.0) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    s = get_settings()
    gem = build_gemini(s)
    insights = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for cg in load_demo(s.demo_file):
        entry = insights.setdefault(cg["id"], {"reviews": {}})
        for r in cg["reviews"]:
            if str(r["id"]) in entry["reviews"]:
                continue  # resume after an interruption
            entry["reviews"][str(r["id"])] = patient(analyze, gem, r["text"]).model_dump()
            OUT.write_text(json.dumps(insights, ensure_ascii=False, indent=1), encoding="utf-8")  # resume point
            time.sleep(pause)
        counts = {a: Counter() for a in ASPECTS}
        for a in entry["reviews"].values():
            for m in a["aspects"]:
                counts[m["aspect"]][m["sentiment"]] += 1
        entry["aspects"] = {a: dict(c) for a, c in counts.items()}
        if "summary" not in entry:
            entry["summary"] = patient(summarize, gem, [r["text"] for r in cg["reviews"]]).model_dump()
        OUT.write_text(json.dumps(insights, ensure_ascii=False, indent=1), encoding="utf-8")
        print(cg["name"], "→", entry["summary"]["headline"], flush=True)
        time.sleep(pause)


if __name__ == "__main__":
    main()
