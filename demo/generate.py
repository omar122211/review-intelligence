"""Generate the demo campgrounds and their reviews (AI-written, fictional — for the demo page only).

All accuracy numbers in this project come from real human reviews (see eval/). These generated reviews
only fill the YelpCamp-style dashboard, because YelpCamp has no real review data.

    python -m demo.generate
"""
from __future__ import annotations

import json
import sys
import time

from pydantic import BaseModel, Field

from reviews.config import get_settings
from reviews.llm import build_gemini

CAMPGROUNDS = [
    {"id": "pine-hollow", "name": "Pine Hollow Campground", "location": "Bolu, Türkiye",
     "profile": "forest sites by a lake, very quiet at night, spotless new shower block, a bit pricey, 20-minute gravel road"},
    {"id": "sunset-dunes", "name": "Sunset Dunes RV Park", "location": "Fethiye, Türkiye",
     "profile": "beachfront with amazing sunsets, crowded and loud in summer (beach bar music late), bathrooms often dirty at peak times, cheap"},
    {"id": "eagle-ridge", "name": "Eagle Ridge Mountain Camp", "location": "Kaçkar Mountains, Türkiye",
     "profile": "remote alpine meadow with stunning views and hiking, very basic facilities (pit toilets, no showers), cold nights, friendly ranger"},
    {"id": "riverside-family", "name": "Riverside Family Camping", "location": "Antalya, Türkiye",
     "profile": "family-friendly with playground and pool, clean and well run, close to town, sites small and close together, kids noisy in the afternoon"},
    {"id": "old-mill", "name": "Old Mill Glamping", "location": "Cappadocia, Türkiye",
     "profile": "glamping tents near balloon launch site, beautiful location, overpriced for what you get, thin tent walls let in road noise, staff slow to fix problems"},
]


class GeneratedReview(BaseModel):
    author: str = Field(description="A fictional first name only")
    stars: int = Field(ge=1, le=5)
    text: str = Field(description="40-140 words, natural camper voice, specific details")


class GeneratedReviews(BaseModel):
    reviews: list[GeneratedReview]


SYSTEM = """You write realistic, varied campground reviews for a software demo. Reviews are fictional.
Mix ratings so they match the campground's real strengths and weaknesses; vary length, tone and focus
(cleanliness, price, location, facilities, noise). Some reviews should be mixed. No real people's names."""


def main(per_campground: int = 18) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    s = get_settings()
    gem = build_gemini(s)
    out = []
    for cg in CAMPGROUNDS:
        prompt = (f"Campground: {cg['name']} ({cg['location']}). What it is like: {cg['profile']}.\n"
                  f"Write {per_campground} reviews.")
        res = gem.generate(system=SYSTEM, prompt=prompt, schema=GeneratedReviews, temperature=0.9)
        reviews = [{"id": i, **r.model_dump()} for i, r in enumerate(res.reviews[:per_campground], 1)]
        out.append({k: cg[k] for k in ("id", "name", "location")} | {"generated_by_ai": True, "reviews": reviews})
        print(cg["name"], len(reviews), "reviews, stars:", [r["stars"] for r in reviews], flush=True)
        time.sleep(4)
    s.demo_file.parent.mkdir(exist_ok=True)
    s.demo_file.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
