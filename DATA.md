# Data

This document describes the datasets and files in this repo so you can understand the pipeline and reproduce results.

## Data sources

- **indian_food.csv** — Indian Food 101 dataset (255 dishes). Public dataset; columns: name, ingredients, diet, flavor_profile, course, state, region.
- **swiggy_cleaned.csv** — Food category labels (e.g. North Indian, Biryani) used only to build category vocabulary for query generation. Not dish-level data.
- **validation_queries.json** — Hand-written query list with category and expected_keywords for validation (see DATA.md layout). No external source.
- **Optional:** `llm_queries*.json` (format: `[{"query": "...", "dish": "..."}]`) can be added to `data/raw/` for extra training pairs; not required. No LLM prompts are stored in the repo (we use programmatic generation as the main source).

## Preprocessing steps (prepare_data.py)

1. **Load & clean dishes** — Read indian_food.csv; map `-1` → missing; parse ingredients to list; lowercase; optional Archana’s Kitchen load with Devanagari filter and name cleaning.
2. **Build indices** — Ingredient → dish names; metadata (diet, flavor, course, region, state) → dish names.
3. **Generate (query, dish_name) pairs** — Exact match, partial name, ingredient (with name-overlap filter), category, cuisine, attribute, occasion, Hinglish templates, synthetic “LLM-style” templates; cap dishes per query to limit label collision.
4. **Augment** — Word reorder (short queries), case variants, random typos (swap/delete/duplicate).
5. **Optional** — Load `llm_queries*.json` from `data/raw/`; filter by catalog and quality (query length, non-dish patterns).
6. **Dedupe** — By (query.lower(), dish_name.lower()).
7. **Split** — Train/val by dish (no query leakage); optional stratify by source. Write dishes.csv, train.csv, val.csv.

## Query generation (no external prompts)

Training queries are generated **programmatically** (no LLM prompts in repo). Patterns: exact dish name + lowercase; partial name (drop word / first / last); ingredient + “dish”/“wala”/“something with X” (only when ingredient appears in dish name); diet+course, flavor+course, region+course; attribute + dish word (e.g. “spicy paneer”); occasion phrases (e.g. “party snack”) mapped to course + name overlap; Hinglish (“kuch X chahiye”, “X banao”); synthetic natural templates per dish. See `prepare_data.py` and DESIGN.md for function list.

## Layout

- **data/raw/** — `validation_queries.json` (used by run_validation_queries). Optional: `llm_queries*.json` for extra training pairs.
- **data/processed/** — Outputs of `prepare_data.py`: `dishes.csv`, `train.csv`, `val.csv`.

## Processed files

- **dishes.csv** — One row per dish. Columns: name, ingredients, diet, flavor, course, state, region, style. Used for ranking (BM25 + DL use rich text: name | diet | course | flavor | region | state | style | ingredients).
- **train.csv** — query, dish_name (positive pairs).
- **val.csv** — query, dish_name (held-out for R@K / MRR).

## Validation set

`data/raw/validation_queries.json` — list of `{ "query", "category", "expected_keywords" }` for assignment categories. Used by `python -m inference.run_validation_queries`.
