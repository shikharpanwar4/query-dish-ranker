# Data

This document describes the datasets and files in this repo so you can understand the pipeline and reproduce results.

## Layout

- **data/raw/** — `validation_queries.json` (used by run_validation_queries). Optional: add `llm_queries*.json` for extra training pairs.
- **data/processed/** — Outputs of `prepare_data.py`: `dishes.csv`, `train.csv`, `val.csv`.

## Dish catalog

**Primary:** `data/indian_food.csv` (Indian Food 101, 255 dishes). Columns: name, ingredients, diet, flavor_profile, course, state, region.

**Optional:** To use Archana’s Kitchen, set `USE_ARCHANAS = True` in `prepare_data.py` and add `IndianFoodDatasetCSV.csv` to `data/` (not included in repo).

**Categories:** `data/swiggy_cleaned.csv` — used for food_type category list (e.g. “North Indian”, “Biryani”) to generate cuisine-style queries.

## Processed files

- **dishes.csv** — One row per dish. Columns: name, ingredients, diet, flavor, course, state, region, style. Used for ranking (BM25 + DL use rich text: name | diet | course | flavor | region | state | style | ingredients).
- **train.csv** — query, dish_name (positive pairs).
- **val.csv** — query, dish_name (held-out for R@K / MRR).

## Query sources

Training pairs come from:

1. Programmatic: exact/partial dish name, cuisine, category, attribute, occasion, Hinglish/synthetic templates, typos and case variants.
2. Optional LLM: if you add `llm_queries*.json` to `data/raw/` (each item: `query`, `dish`), they are loaded and merged. Not required.

## Validation set

`data/raw/validation_queries.json` — list of `{ "query", "category", "expected_keywords" }` for assignment categories. Used by `python -m inference.run_validation_queries`.
