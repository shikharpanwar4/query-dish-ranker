# Design & data flow

Short reference for functions, classes, and high-level data flow. For reproduction and usage see README and CHALLENGE.

---

## High-level data flow

### 1. Data preparation

```
indian_food.csv + swiggy_cleaned.csv
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  prepare_data.py main()                                        │
│  • Load & clean dishes (load_indian_food; optional Archana’s)  │
│  • Build ingredient + metadata indices                         │
│  • Generate (query, dish_name) pairs (exact, partial,          │
│    ingredient, category, cuisine, attribute, occasion,         │
│    hinglish, synthetic LLM-style)                              │
│  • Augment (reorder, case, typos)                              │
│  • Optional: load llm_queries*.json                            │
│  • Dedupe → split train/val by dish → save                     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
dishes.csv, train.csv, val.csv  (in data/processed/)
```

### 2. Training

```
train.csv + dishes.csv (for rich dish text)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  training/train.py train()                                     │
│  • QueryDishDataset: (query, dish) pairs; query typo at load  │
│  • DualEncoderScorer: shared TextEncoder (trigram → embed)    │
│  • InfoNCE loss (in-batch negatives)                           │
│  • AdamW + warmup + cosine decay; eval = full-catalog R@K      │
│  • Save best checkpoint by val R@5                             │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
checkpoints/best_model.pt
```

### 3. Inference

```
query string + dishes.csv (names + rich text)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  inference/query.py hybrid_rank()                              │
│  • Query norm (optional chaat expand)                          │
│  • Lexical: fuzzy match query terms vs dish names              │
│  • BM25: score query vs dish display text (name|diet|…)        │
│  • DL: encode query + dot with precomputed dish embeddings     │
│  • Min-max norm each, weighted sum (0.4 lex, 0.35 bm25, 0.25 dl)│
│  • Return (index, score) sorted desc                           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
top-K dish names (and optionally scores)
```

---

## Modules and symbols (brief)

### data/prepare_data.py

| Symbol | One-liner |
|--------|-----------|
| `load_indian_food` | Load indian_food.csv; -1→None, ingredients→list, lowercase. |
| `extract_swiggy_food_categories` | Unique food_type from swiggy CSV. |
| `build_ingredient_index` | ingredient → list of dish names. |
| `build_metadata_index` | Group dishes by diet/flavor/course/region/state. |
| `clean_recipe_name` | Strip suffix/parens from Archana’s recipe names; cap 8 words. |
| `clean_ingredient_str` | Parse Archana’s ingredient string → list of nouns. |
| `load_archanas_kitchen` | Load Archana’s CSV; filter Devanagari, clean, map cuisine→region. |
| `generate_exact_match_pairs` | (dish_name, dish_name) and lowercase variant. |
| `generate_partial_name_pairs` | Partial name queries (drop word / first / last); 2+ words. |
| `generate_ingredient_query_pairs` | Ingredient queries; only pairs where ingredient in dish name. |
| `generate_category_query_pairs` | Diet+course, flavor+course, region+course, swiggy categories. |
| `generate_cuisine_query_pairs` | Cuisine+course and cuisine+suffix (Archana’s cuisine field). |
| `generate_attribute_query_pairs` | Attribute + dish-word, e.g. spicy paneer, veg biryani. |
| `generate_occasion_query_pairs` | e.g. party snack, sweet dessert → matching course + name word. |
| `generate_hinglish_query_pairs` | Hinglish templates with dish-name word. |
| `generate_synthetic_llm_style_pairs` | LLM-style natural queries per dish (Hinglish/occasion/attr). |
| `inject_typo` | One or more char typos (swap/delete/duplicate) in a word. |
| `augment_pairs` | Add reorder, case, and static typo variants. |
| `load_llm_pairs` | Load llm_queries*.json; alias map; catalog filter; quality filter. |
| `deduplicate_pairs` | Dedupe by (q.lower(), d.lower()). |
| `split_train_val` | Split by dish (no leakage); optional stratify by source. |
| `save_pairs` | Write (query, dish_name) CSV. |
| `save_dish_catalog` | Write dishes CSV with name, ingredients, diet, flavor, course, state, region, style. |
| `print_stats` | Log pair count and unique query/dish counts. |
| `main` | Run full pipeline: load → generate → augment → LLM → dedupe → split → save. |

### model/

| Symbol | One-liner |
|--------|-----------|
| **tokenizer.py** | |
| `CharTrigramTokenizer` | Stateless: text → trigram IDs via hashing; pad_id=0. |
| `encode` | Single text → list of bucket IDs (pad/trunc to max_trigrams). |
| `encode_batch` | List of texts → list of ID lists. |
| **encoder.py** | |
| `TextEncoder` | Trigram IDs → L2-normalized embed (embedding → mean pool → proj → norm). |
| `forward` | [B, max_trigrams] → [B, embed_dim]. |
| **scorer.py** | |
| `DualEncoderScorer` | Wraps tokenizer + single shared TextEncoder for query and dish. |
| `encode` | List of texts → [N, embed_dim] (used for both query and dish). |
| `forward` | (query_ids, dish_ids) → similarity logits. |
| `score_pairs` | Batch (query, dish) → scores. |
| `score_one_to_many` | One query vs many dishes → scores. |

### training/

| Symbol | One-liner |
|--------|-----------|
| **dataset.py** | |
| `get_dish_display_text` | name \| diet \| course \| flavor \| region \| state \| style \| first 5 ingredients. |
| `_inject_typo_online` | One random typo in a word (swap/delete/duplicate). |
| `QueryDishDataset` | (query, dish) pairs; query typo at getitem; returns query_ids, dish_ids, label. |
| `create_dataloaders` | Train/val DataLoaders from train/val CSVs + optional dish catalog. |
| **loss.py** | |
| `InfoNCELoss` | In-batch negatives; forward(logits, labels) → loss. |
| **train.py** | |
| `compute_full_catalog_recall` | Rank each val query vs full catalog; return R@K dict. |
| `infer_query_type` | Heuristic: exact / partial / category / attribute / hinglish / other. |
| `compute_per_type_recall` | R@K per inferred query type. |
| `print_per_type_report` | Print per-type recall table. |
| `get_lr` | Warmup + cosine decay schedule. |
| `train` | Train loop: DataLoaders, DualEncoderScorer, InfoNCE, AdamW; validate by R@5; save best. |
| `save_checkpoint` | Save model, tokenizer config, loss, history, epoch. |
| `load_checkpoint` | Load checkpoint; return (model, config). |

### inference/

| Symbol | One-liner |
|--------|-----------|
| **query.py** | |
| `_collapse_repeats` | e.g. daal→dal, fryy→fry. |
| `tokenize_for_lexical` | Lowercase, split non-alnum, drop short tokens. |
| `_edit_distance` | Levenshtein distance. |
| `lexical_score` | Query terms in dish name (exact + fuzzy edit dist 1). |
| `_query_for_scoring` | Append " chaat" if query has chaat-style terms. |
| `rank_lexical` | Lexical-only ranking: (index, score) desc. |
| `_normalize_scores` | Min-max to [0,1]. |
| `hybrid_rank` | Lexical + BM25 + DL; norm each; weighted sum; return (idx, score) desc. |
| `main` | CLI: single query, optional --lexical/--compare/--hybrid, --top N. |
| **bm25.py** | |
| `_collapse_repeats` | Same as query.py. |
| `tokenize` | Lowercase, split, drop empty. |
| `BM25` | Build index from doc list; score(query) → list of scores. |
| **eval_hybrid.py** | |
| `main` | Load val + dishes; hybrid_rank each; compute R@1/R@5/R@10, MRR. |
| **latency_benchmark.py** | |
| `main` | End-to-end latency + breakdown (query_norm, lexical, BM25, DL, combine+sort). |
| **run_validation_queries.py** | |
| `_keyword_hit_at_k` | True if any top-K result contains any expected keyword. |
| `main` | Run validation_queries.json through hybrid_rank; print metrics + optional per-query. |

---

## File roles (quick ref)

| Path | Role |
|------|------|
| `data/indian_food.csv` | Main dish catalog (255). |
| `data/swiggy_cleaned.csv` | Category vocabulary for query generation. |
| `data/processed/dishes.csv` | Clean catalog + style; input to inference. |
| `data/processed/train.csv`, `val.csv` | (query, dish_name) for training. |
| `data/raw/validation_queries.json` | Category queries for run_validation_queries. |
| `checkpoints/best_model.pt` | Best checkpoint by val R@5; used by inference/eval. |
