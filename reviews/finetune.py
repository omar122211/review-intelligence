"""Fine-tune a small transformer (MiniLM-L6, 22M parameters) for overall sentiment.

    python -m reviews.finetune              # 2 epochs, ~1 h on a laptop CPU
    python -m reviews.finetune --limit 256  # smoke run

Writes models/sentiment-minilm/ (fp16 weights + tokenizer) and eval/finetune.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from reviews.config import ROOT, get_settings
from reviews.labels import SENTIMENTS

MAX_TOKENS = 256


def batches(df: pd.DataFrame, tok, batch: int, shuffle: bool) -> DataLoader:
    enc = tok(df["review"].tolist(), truncation=True, max_length=MAX_TOKENS)
    items = [{"input_ids": i, "attention_mask": m, "labels": SENTIMENTS.index(y)}
             for i, m, y in zip(enc["input_ids"], enc["attention_mask"], df["sentiment"])]
    return DataLoader(items, batch_size=batch, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tok))


@torch.inference_mode()
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for b in loader:
        ys.append(b.pop("labels").numpy())
        ps.append(model(**b).logits.argmax(-1).numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.manual_seed(7)
    s = get_settings()
    d = s.data_dir / "splits"
    data = {k: pd.read_parquet(d / f"{k}.parquet") for k in ("train", "val", "test")}
    if args.limit:
        data = {k: v.head(args.limit) for k, v in data.items()}

    tok = AutoTokenizer.from_pretrained(s.embed_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        s.embed_model, num_labels=3, id2label=dict(enumerate(SENTIMENTS)), label2id={l: i for i, l in enumerate(SENTIMENTS)})
    train, val, test = (batches(data["train"], tok, args.batch, True), batches(data["val"], tok, 64, False),
                        batches(data["test"], tok, 64, False))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    history, best = [], (-1.0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for step, b in enumerate(train, 1):
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train)} loss {total / step:.4f}", flush=True)
        y, p = predict(model, val)
        f1 = f1_score(y, p, average="macro")
        history.append({"epoch": epoch, "train_loss": round(total / len(train), 4), "val_macro_f1": round(f1, 4),
                        "minutes": round((time.time() - started) / 60, 1)})
        print(history[-1], flush=True)
        if f1 > best[0]:
            best = (f1, {k: v.detach().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    t0 = time.time()
    y, p = predict(model, test)
    ms = (time.time() - t0) / len(y) * 1000
    result = {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4),
              "ms_per_review": round(ms, 1), "history": history, "epochs": args.epochs, "max_tokens": MAX_TOKENS}
    print("test", result, flush=True)

    out = s.models_dir / "sentiment-minilm"
    model.half().save_pretrained(out, safe_serialization=True)  # fp16 → ~45 MB
    tok.save_pretrained(out)
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "finetune.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
