# Search Ranking — Query–Dish Matcher

Dual-encoder retrieval for food search: given a query (English or Hinglish), rank dishes by relevance. Combines **lexical** (fuzzy token match), **BM25**, and a small **bi-encoder** (char trigrams → shared encoder, InfoNCE). No reranker in the shipped pipeline; see DESIGN.md for what we tried and why.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

## Setup

```bash
pip install -r requirements.txt
```

## Data (if training from scratch)

1. **Prepare data** — builds dish catalog and train/val splits from CSVs (and optional LLM query JSONs):

   ```bash
   python data/prepare_data.py
   ```

   Outputs: `data/processed/dishes.csv`, `train.csv`, `val.csv`. See [DATA.md](DATA.md).

2. **Train** — bi-encoder with InfoNCE; best checkpoint by val R@5:

   ```bash
   python training/train.py
   ```

   Saves `checkpoints/best_model.pt` and `checkpoints/final_model.pt`.

## Inference

- **Single query (hybrid):**

  ```bash
  python -m inference.query "paneer tikka" --hybrid --top 5
  ```

- **Eval on val set** (R@1, R@5, R@10, R@50, MRR):

  ```bash
  python -m inference.eval_hybrid
  ```

  Options: `--mode bi|hybrid`, `--weights w_lex w_bm25 w_dl`, `--per-type`, `--tune-weights`.

- **Latency** (500 dishes, CPU, precomputed embeddings):

  ```bash
  python -m inference.latency_benchmark
  ```

- **Validation queries** (category coverage):

  ```bash
  python -m inference.run_validation_queries --metrics-only
  python -m inference.run_validation_queries --top 5
  ```

## Model

- **Trained weights:** `checkpoints/best_model.pt` (bi-encoder; include in repo so eval runs without training).
- **Inference:** `python -m inference.query "<query>" --hybrid --top K` (lexical + BM25 + DL).

## Results

Metrics on the full validation set (~4.8k pairs, ~2k dishes). Default hybrid weights (0.10, 0.45, 0.45) — lexical, BM25, DL.

| Metric | Bi-encoder only | Hybrid (default) | Hybrid (tuned weights) |
|--------|-----------------|------------------|-------------------------|
| R@1    | ~0.28           | **0.49**         | **0.51**                |
| R@5    | ~0.64           | **0.72**         | **0.75**                |
| R@10   | ~0.80           | **0.81**         | **0.82**                |
| R@50   | —               | **0.94**         | **0.95**                |
| MRR    | ~0.43           | **0.60**         | **0.62**                |

Tuned weights (grid search on val MRR): e.g. lexical=0.25, BM25=0.50, DL=0.25. Run `python -m inference.eval_hybrid --tune-weights` to reproduce.

**Latency (500 dishes, CPU):** mean ~25–30 ms, P50 ~15 ms, P99 &lt;130 ms. Target &lt;100 ms (mean) is met on typical runs; lexical over 500 names is the main cost. Model size &lt;20 MB.

Details of training trials, tradeoffs, and why we don’t ship the reranker are in [DESIGN.md](DESIGN.md).

## Five qualitative examples

Run: `python -m inference.query "<query>" --hybrid --top 5`

| Query | Top result(s) |
|-------|----------------|
| butter chicken | Butter chicken |
| spicy paneer | Paneer tikka masala / Shahi paneer |
| dal fryy (misspelled) | Dal-based dishes (Daal, Dal makhani, etc.) |
| kuch meetha chahiye (Hinglish) | Gulab jamun, Kaju katli, sweet desserts |
| party snack | Samosa, Dahi vada, chaat-style snacks |

## Reproducing

With `checkpoints/best_model.pt` in the repo:

```bash
pip install -r requirements.txt
python -m inference.eval_hybrid
python -m inference.latency_benchmark
python -m inference.run_validation_queries --metrics-only
```

To train from scratch: `python data/prepare_data.py` then `python training/train.py`.

## Language

English and Hinglish (Roman script) are supported. Devanagari is not in the training data; Romanized Hindi/Hinglish only.
