# Results

All numbers are on **real reviews never seen in training** (TripAdvisor, reviewers' own star ratings as labels; classes balanced, so always guessing "positive" would score 33 %).

## Overall sentiment — full test set (3,000 reviews)

| Approach | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression | 78.0% | 0.779 |
| MiniLM embeddings + logistic regression | 73.4% | 0.734 |
| **Fine-tuned MiniLM** | **79.4%** | **0.794** |

## Same 300 test reviews, including Gemini

| Approach | Accuracy | Macro F1 | Time per review |
|---|---|---|---|
| TF-IDF + logistic regression | 75.3% | 0.753 | – |
| MiniLM embeddings + logistic regression | 70.3% | 0.702 | – |
| Fine-tuned MiniLM | 78.3% | 0.782 | 46 ms |
| Gemini zero-shot | 77.3% | 0.743 | 954 ms |

## Aspects

Gemini lists only the aspects a review talks about, with a quote. When it calls an aspect positive or negative, does the reviewer's own star rating for that aspect agree? (Only clear cases: aspect stars ≠ 3.)

| Aspect | Mentioned in | Agreement with reviewer's aspect stars |
|---|---|---|
| Cleanliness | 157/300 reviews | 113/117 (97%) |
| Value for money | 154/300 reviews | 98/108 (91%) |
| Location | 214/300 reviews | 175/180 (97%) |
| Sites & facilities | 276/300 reviews | 160/180 (89%) |
| Quiet & sleep | 106/300 reviews | 68/84 (81%) |

The local aspect heads (MiniLM embeddings + logistic regression) predict every aspect for every review, mentioned or not — macro F1 on the full test set:

| Aspect | Embeddings | TF-IDF |
|---|---|---|
| Cleanliness | 0.623 | 0.653 |
| Value for money | 0.635 | 0.656 |
| Location | 0.469 | 0.507 |
| Sites & facilities | 0.645 | 0.699 |
| Quiet & sleep | 0.607 | 0.625 |
