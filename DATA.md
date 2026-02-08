# Data

Sources, how training pairs are produced, and preprocessing.

## Data sources

| Source | Role |
|--------|------|
| **indian_food.csv** | Core dish catalog (e.g. Indian Food 101). Columns: name, ingredients, diet, flavor_profile, course, state, region. |
| **swiggy_cleaned.csv** | Food category labels (North Indian, Biryani, etc.) used only as vocabulary for query generation. |
| **swiggy_all_menus_bangalore.csv** | Optional. More dish names + categories; merged into catalog with minimal metadata. |
| **validation_queries.json** | Hand-written queries with category and expected_keywords for assignment-style validation. |
| **llm_queries*.json** (optional) | LLM-generated queries in `data/raw/`. Two formats: `[{"query","dish"}]` or `[{"query","category","expected_keywords"}]`. Keywords are resolved to catalog dishes. |

## Preprocessing (prepare_data.py)

1. **Load & clean** — Read dish CSVs; normalize missing values; parse ingredients; lowercase where needed. Optional Archana’s Kitchen CSV with Devanagari filter.
2. **Merge catalog** — Indian food + Swiggy (or other) dishes; dedupe by normalized name; keep richest metadata per name.
3. **Indices** — Ingredient → dishes; metadata (diet, flavor, course, region, state) → dishes for query generation.
4. **Generate (query, dish_name) pairs** — Exact match, partial name, ingredient (only when ingredient appears in dish name), category, cuisine, attribute, occasion, Hinglish templates, synthetic “LLM-style” templates. Cap labels per query to limit noise.
5. **Augment** — Title/upper case (English); word reorder for short queries; limited typos (e.g. 10%, fixed seed) on English.
6. **Optional LLM** — Load `llm_queries*.json` from `data/raw/`; filter by catalog and simple quality (length, no recipe-style phrases).
7. **Dedupe** — By `(query.lower(), dish_name.lower())`.
8. **Split** — Train/val by dish (no leakage). Write `dishes.csv`, `train.csv`, `val.csv` to `data/processed/`.

## Query generation (no external API in repo)

Training queries are **programmatic** (no LLM calls in the script). Patterns:

- Exact dish name + lowercase.
- Partial name: drop word, first word, last word (2+ word names).
- Ingredient + “dish”/“wala”/“something with X” only when ingredient is in dish name.
- Diet+course, flavor+course, region+course; attribute + dish word (“spicy paneer”); occasion phrases (“party snack”) mapped to course + name overlap.
- Hinglish: “kuch X chahiye”, “X banao”, etc., with dish-name word.
- Synthetic natural templates per dish.

See `prepare_data.py` and DESIGN.md for the function list.

## Optional LLM query generation

To add more diverse queries, you can use an LLM and save JSON under `data/raw/`:

- **Format 1:** `[{"query": "...", "dish": "..."}]` — dish must be in catalog.
- **Format 2:** `[{"query": "...", "category": "...", "expected_keywords": ["kw1", "kw2"]}]` — keywords are matched to catalog (name/ingredients/category).

Categories to cover: exact_dish_name, misspelled_dish_name, cuisine_type, dietary_preference, occasion_based, attribute_based, hinglish, vague_exploratory. Keep queries short (1–6 words typical), Roman script only, no recipe/how-to. Validate with:

```bash
python data/scripts/validate_llm_queries.py data/raw/llm_queries.json
```

## Layout

- **data/raw/** — `validation_queries.json`; optional `llm_queries*.json`.
- **data/processed/** — `dishes.csv`, `train.csv`, `val.csv` (outputs of `prepare_data.py`).

## Processed files

- **dishes.csv** — One row per dish: name, ingredients, diet, flavor, course, state, region, style (and category if present). Used for ranking (BM25 and DL use a concatenated text: name | diet | course | … | ingredients snippet).
- **train.csv**, **val.csv** — Columns `query`, `dish_name` (positive pairs).
