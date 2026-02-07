# Challenge 2B — Proof of Requirements

This document is for reviewers: it states the reported numbers and how you can reproduce them.

## Summary

| Requirement | Target | Result |
|-------------|--------|--------|
| End-to-end latency (CPU, 500 items) | < 100 ms | **PASS** (see §1 for mean/P50/P99 and breakdown) |
| Model size | < 20 MB | **~1.47 MB** (386K params) |
| Eval (val set) | — | R@1 **0.60**, R@5 **0.82**, R@10 **0.90**, MRR **0.71** |
| Query types | Hindi, English, Hinglish | English + Hinglish (Roman). Hindi = Romanized only. |

---

## 1. End-to-end latency and breakdown

**End-to-end:** One query in → ranked top-K out (500 dishes, CPU, precomputed dish embeddings + BM25 index). Goal: optimize this number.

**How to run:** From the repo root, `python -m inference.latency_benchmark`

**Output (representative):**

```
Params: 386,448  (~1.47 MB)
Requirement: <20 MB  →  PASS

End-to-end latency: 1 query → ranked top-K (500 dishes, CPU, precomputed dish embs)
  End-to-end mean:  ~26 ms
  End-to-end P50:   ~15 ms
  End-to-end P99:   ~127 ms
  Requirement: <100 ms (CPU)  →  PASS

Breakdown (mean ms per query, % of end-to-end):
  query norm (chaat expand)         ~0.4 ms   (~1%)
  lexical (500 × fuzzy match)      ~13 ms   (~51%)   ← main cost
  BM25 (500 docs)                   ~3 ms   (~12%)
  DL (encode query + dot)           ~8 ms   (~32%)
  normalize + combine + sort        ~0.8 ms   (~3%)
```

Use the breakdown to prioritize: lexical (fuzzy over 500 names) and DL (query encode + dot) dominate; BM25 and combine/sort are smaller.

---

## 2. Model size and checkpoint location

- **Checkpoint:** `checkpoints/best_model.pt` — used by all inference and eval scripts. If present in the repo, you can run eval and latency without training.
- Parameters: **386,448**; size **~1.47 MB** (float32). Requirement **< 20 MB** → **PASS**.

---

## 3. Evaluation metrics (val set)

Hybrid ranker: lexical (0.4) + BM25 (0.35) + DL (0.25). Correct dish = exact match to val `dish_name`.

**How to run:** `python -m inference.eval_hybrid`

**Output (representative):**

```
Valid pairs:  701
Catalog:     255 dishes

  R@1:  0.6034
  R@5:  0.8231
  R@10: 0.8959
  MRR:  0.7053
```

---

## 4. Validation queries (category coverage)

Queries by category: exact dish name, misspelled, cuisine, dietary, occasion, attribute, Hinglish, vague. Metric: **keyword hit @ 3** — at least one of top-3 results contains at least one expected keyword.

**How to run:** `python -m inference.run_validation_queries --metrics-only`

**Output (representative):**

```
Queries with expected_keywords: 23 / 24
Keyword hit @ 3:  23/23 = 100.0%

Per-category keyword hit @ 3:
  attribute_based: 3/3 = 100%
  cuisine_type: 3/3 = 100%
  dietary_preference: 4/4 = 100%
  exact_dish_name: 2/2 = 100%
  hinglish: 3/3 = 100%
  misspelled_dish_name: 3/3 = 100%
  occasion_based: 3/3 = 100%
  vague_exploratory: 2/2 = 100%
```

Query list: `data/raw/validation_queries.json`.

---

## 5. How to reproduce

**If you have `checkpoints/best_model.pt` in the repo (recommended):**

1. `pip install -r requirements.txt`
2. From repo root: `python -m inference.latency_benchmark`, `python -m inference.eval_hybrid`, `python -m inference.run_validation_queries --metrics-only`

**If you want to train from scratch:**

1. `pip install -r requirements.txt`
2. `python data/prepare_data.py`
3. `python training/train.py`
4. Then run the three inference commands above. Checkpoints are written to `checkpoints/`.
