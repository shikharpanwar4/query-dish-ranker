# Search Ranking (Query–Dish Matcher)

This repo implements a dual-encoder search ranker for query–dish relevance: hybrid of lexical (fuzzy) + BM25 + learned embeddings. It supports English, Hinglish, and Romanized Hindi.

## Requirements

- Python 3.10+
- See `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
```

## Data (if training from scratch)

1. **Prepare data** (dish catalog + train/val splits):

   ```bash
   python data/prepare_data.py
   ```

   Reads `data/indian_food.csv`, `data/swiggy_cleaned.csv`, and optional `data/raw/llm_queries*.json`. Writes `data/processed/dishes.csv`, `train.csv`, `val.csv`. See [DATA.md](DATA.md).

2. **Train** (dual-encoder + InfoNCE):

   ```bash
   python training/train.py
   ```

   Saves best checkpoint to `checkpoints/best_model.pt` (by val R@5).

## Inference and benchmarks

- **Single query (hybrid):**

  ```bash
  python -m inference.query "paneer tikka" --hybrid --top 5
  ```

- **Eval on val set (R@1, R@5, R@10, MRR):**

  ```bash
  python -m inference.eval_hybrid
  ```

- **Latency benchmark** (500 dishes, CPU, precomputed embeddings):

  ```bash
  python -m inference.latency_benchmark
  ```

- **Validation queries** (assignment categories: exact, misspelled, cuisine, dietary, occasion, attribute, Hinglish, vague):

  ```bash
  python -m inference.run_validation_queries --metrics-only
  python -m inference.run_validation_queries --top 5
  ```

## Project structure

```
search-ranking/
├── data/
│   ├── indian_food.csv          # Main dish catalog (Indian Food 101)
│   ├── swiggy_cleaned.csv       # Category vocabulary
│   ├── raw/                     # LLM query JSONs, validation_queries.json
│   ├── processed/               # dishes.csv, train.csv, val.csv
│   └── prepare_data.py
├── model/                       # Tokenizer, encoder, dual-encoder scorer
├── training/                    # Dataset, loss, train loop
├── inference/                   # query.py, bm25, eval_hybrid, latency_benchmark, run_validation_queries
├── checkpoints/                 # best_model.pt (for eval/latency; see Reproducing results)
├── CHALLENGE.md                 # Reported metrics, latency breakdown, how to reproduce
├── DATA.md                      # Data layout and sources
├── DESIGN.md                    # Data flow, and brief function/class reference
└── requirements.txt
```

## Reproducing results

If the repo includes `checkpoints/best_model.pt`, you can run eval and benchmarks without training:

```bash
pip install -r requirements.txt
python -m inference.eval_hybrid
python -m inference.latency_benchmark
python -m inference.run_validation_queries --metrics-only
```

To train from scratch: run `python data/prepare_data.py`, then `python training/train.py`. See [CHALLENGE.md](CHALLENGE.md) for the reported metrics and how they were produced.

## Language support

- **English & Hinglish** (Roman script): full support.
- **Hindi (Devanagari)**: not in training data; only Romanized Hindi/Hinglish is supported. For Devanagari, add transliteration before scoring.
