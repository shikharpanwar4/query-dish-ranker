# Design and tradeoffs

High-level flow, what we built, what we tried, and what we’d do next.

## Data flow

1. **Data** — `prepare_data.py` loads dish CSVs and optional LLM JSONs, builds (query, dish_name) pairs (exact, partial, ingredient, category, Hinglish, etc.), augments, dedupes, splits by dish → `dishes.csv`, `train.csv`, `val.csv`.
2. **Training** — `training/train.py` trains a bi-encoder (char trigrams → shared encoder, InfoNCE). Validation = full-catalog R@1/R@5/R@10; best checkpoint by R@5. Saves `best_model.pt`, `final_model.pt`.
3. **Inference** — One stage: **hybrid_rank** = lexical (fuzzy token match) + BM25 + bi-encoder; scores min-max normalized and combined with fixed weights. No reranker in the shipped pipeline.

## Why this design

- **Lexical + BM25 + DL** — Lexical handles typos and exact tokens; BM25 adds IDF over the catalog; the bi-encoder captures semantic similarity (e.g. “meetha” → sweets). Weight tuning on val (e.g. 0.10, 0.45, 0.45 or 0.25, 0.50, 0.25) improves MRR over defaults.
- **Single-stage only** — We experimented with a cross-encoder reranker on top-50. It was trained with random negatives; at eval it had to beat 49 hard negatives from Stage 1 and consistently ranked the correct dish below incorrect ones (e.g. ~3% correct at #1). Fixing that would require training the reranker with Stage-1 hard negatives (and possibly listwise loss). We left the reranker out of the submission and ship the strong hybrid Stage 1 only.
- **Small bi-encoder** — Char trigrams + feature hashing (no vocab file), shared encoder for query and dish, L2-normalized embeddings, InfoNCE. Fits latency and size constraints; precomputed dish embeddings keep inference fast.

## Metrics we tracked

From runs on the full val set and 500-pair subsets:

- **Bi-encoder only:** R@5 ~0.62–0.64, MRR ~0.41–0.43 (full catalog).
- **Hybrid (default weights):** R@1 ~0.49, R@5 ~0.72, R@10 ~0.81, R@50 ~0.94, MRR ~0.60.
- **Hybrid (tuned weights):** MRR ~0.62, R@1 ~0.51, R@5 ~0.75, R@10 ~0.82.
- **Latency (500 dishes, CPU):** mean ~25–30 ms, P50 ~15 ms, P99 &lt;130 ms; target &lt;100 ms (mean) met in typical runs. Breakdown: lexical over 500 names is the largest share; then DL (query encode + dot with precomputed dish embs); BM25 and combine/sort are smaller.

Training trials (abbreviated): we tried larger vs smaller capacity, with and without hard negatives (precomputed from Stage 1). Best bi-encoder val R@5 was ~0.64 (1M params, in-batch negatives only). Hard negatives did not help in our setup. Hybrid consistently beat bi-encoder-only on R@1/R@5/MRR.

## Module overview

- **data/prepare_data.py** — Load CSVs, build indices, generate and augment pairs, optional LLM load, dedupe, split, save.
- **model/** — `tokenizer.py` (trigrams + hashing), `encoder.py` (embed → mean pool → proj → L2 norm), `scorer.py` (DualEncoderScorer: one encoder for query and dish).
- **training/** — `dataset.py` (QueryDishDataset, optional typo augmentation), `loss.py` (InfoNCE), `train.py` (loop, full-catalog val, per-query-type report, checkpointing).
- **inference/** — `query.py` (lexical_score, hybrid_rank), `bm25.py` (BM25 over dish texts), `eval_hybrid.py` (bi vs hybrid, optional per-type and weight tuning), `latency_benchmark.py`, `run_validation_queries.py`.

## Next steps (evolution)

1. **Reranker (if needed)** — Train cross-encoder on Stage-1 top-K with hard negatives (and optionally listwise loss); validate with “% correct at #1 in top-50” and tune checkpoint selection.
2. **Data** — More Hinglish and “other” query types; optional LLM-generated queries with strict validation.
3. **Latency** — Reduce lexical cost (e.g. smaller candidate set from DL first, then lexical/BM25 on that set; or faster fuzzy match).
4. **Model** — Try slightly larger embed dim or more buckets if catalog grows, with the same single-stage hybrid design.
