"""Build the train / validation / test sets from 200k real TripAdvisor reviews.

Every review comes with the reviewer's own star ratings — overall and per aspect (cleanliness, value,
location, rooms, sleep quality) — which serve as human labels.

- Classes are balanced (equal negative / neutral / positive) so accuracy can't be won by always
  saying "positive" (77 % of the raw reviews are 4-5 stars).
- The splits never share a reviewer, so the model can't score points by recognising a writing style.

    python -m reviews.data        # → data/splits/{train,val,test}.parquet
"""
from __future__ import annotations

import pandas as pd
from huggingface_hub import hf_hub_download

from reviews.config import get_settings
from reviews.labels import ASPECTS, stars_to_sentiment

REPO, FILE = "jniimi/tripadvisor-review-rating", "data/train-00000-of-00001.parquet"
PER_CLASS = {"train": 4000, "val": 500, "test": 1000}
MAX_CHARS = 2000


def load_raw() -> pd.DataFrame:
    s = get_settings()
    df = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset", cache_dir=s.data_dir / "hf-cache"))
    df = df[df["text"].str.len().between(80, 6000)].copy()  # skip one-liners and essays
    df["review"] = (df["title"].fillna("").str.strip() + ". " + df["text"].str.strip()).str.slice(0, MAX_CHARS)
    df["sentiment"] = df["overall"].map(stars_to_sentiment)
    for aspect, col in ASPECTS.items():
        df[f"{aspect}_stars"] = df[col]
    return df


def main() -> None:
    s = get_settings()
    df = load_raw()
    # split by reviewer first (80/5/15 of reviewers), then balance classes inside each split
    users = df["user_id"].drop_duplicates().sample(frac=1.0, random_state=7).tolist()
    n = len(users)
    groups = {"train": set(users[: int(0.8 * n)]), "val": set(users[int(0.8 * n): int(0.85 * n)]), "test": set(users[int(0.85 * n):])}
    out = s.data_dir / "splits"
    out.mkdir(parents=True, exist_ok=True)
    keep = ["review", "overall", "sentiment"] + [f"{a}_stars" for a in ASPECTS]
    for split, members in groups.items():
        part = df[df["user_id"].isin(members)]
        part = pd.concat([part[part["sentiment"] == c].sample(n=min(PER_CLASS[split], (part["sentiment"] == c).sum()), random_state=7)
                          for c in sorted(part["sentiment"].unique())])
        part = part.sample(frac=1.0, random_state=7)[keep].reset_index(drop=True)
        part.to_parquet(out / f"{split}.parquet")
        print(split, len(part), part["sentiment"].value_counts().to_dict())


if __name__ == "__main__":
    main()
