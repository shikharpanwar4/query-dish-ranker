"""
Run validation queries (assignment categories) through the hybrid ranker.

Validates coverage of: exact dish names, misspelled, cuisine, dietary, occasion,
attribute-based, Hinglish, vague/exploratory. Uses data/raw/validation_queries.json.

Eval metrics (keyword-based; no ground-truth dish IDs in JSON):
  - Keyword hit @ K: % of queries where ≥1 of top-K dishes contains ≥1 expected_keyword.
  - Per-category keyword hit @ K: same, broken down by category.

Usage:
  python -m inference.run_validation_queries
  python -m inference.run_validation_queries --top 5
  python -m inference.run_validation_queries --metrics-only   # print only metrics summary
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.train import load_checkpoint
from training.dataset import get_dish_display_text
from inference.query import hybrid_rank


def _keyword_hit_at_k(
    top_dish_names: list[str],
    top_dish_texts: list[str],
    expected_keywords: list[str],
) -> bool:
    """True if any of the top dishes (name or full text) contains any expected keyword (case-insensitive)."""
    if not expected_keywords:
        return True  # nothing to check
    keys = {k.lower() for k in expected_keywords}
    for name, text in zip(top_dish_names, top_dish_texts):
        combined = f"{name} {text}".lower()
        if any(k in combined for k in keys):
            return True
    return False


def main():
    data_dir = ROOT / "data"
    validation_path = data_dir / "raw" / "validation_queries.json"
    dishes_csv = data_dir / "processed" / "dishes.csv"
    checkpoint_path = ROOT / "checkpoints" / "best_model.pt"

    if not validation_path.exists():
        print(f"Missing {validation_path}. Create it with category/expected_keywords per assignment format.")
        sys.exit(1)
    if not dishes_csv.exists() or not checkpoint_path.exists():
        print("Run prepare_data and training first.")
        sys.exit(1)

    with open(validation_path, encoding="utf-8") as f:
        queries = json.load(f)

    dishes = []
    with open(dishes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dishes.append(dict(row))
    dish_names = [d["name"] for d in dishes]
    dish_display_texts = [get_dish_display_text(d) for d in dishes]

    print("Loading model...")
    model, _ = load_checkpoint(str(checkpoint_path), device="cpu")
    top_k = 3
    if "--top" in sys.argv:
        i = sys.argv.index("--top")
        if i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
    metrics_only = "--metrics-only" in sys.argv

    # Collect results and compute keyword-hit metrics
    results = []
    for item in queries:
        q = item.get("query", "").strip()
        cat = item.get("category", "other")
        expected = item.get("expected_keywords", [])
        if not q:
            continue
        ranked = hybrid_rank(q, dish_names, dish_display_texts, model=model)
        top_idxs = [idx for idx, _ in ranked[:top_k]]
        top_names = [dish_names[i] for i in top_idxs]
        top_texts = [dish_display_texts[i] for i in top_idxs]
        hit = _keyword_hit_at_k(top_names, top_texts, expected)
        results.append({"query": q, "category": cat, "expected_keywords": expected, "top_names": top_names, "hit": hit})

    # Eval metrics: keyword hit @ K (only for queries with non-empty expected_keywords)
    with_keys = [r for r in results if r["expected_keywords"]]
    n_with = len(with_keys)
    hits_total = sum(1 for r in with_keys if r["hit"])
    kw_hit_rate = (hits_total / n_with * 100) if n_with else 0.0
    by_cat = defaultdict(list)
    for r in with_keys:
        by_cat[r["category"]].append(r["hit"])

    # Print metrics summary
    print("\n" + "=" * 70)
    print("EVAL METRICS (validation set, keyword-based)")
    print("=" * 70)
    print(f"  Queries with expected_keywords: {n_with} / {len(results)}")
    print(f"  Keyword hit @ {top_k}:  {hits_total}/{n_with} = {kw_hit_rate:.1f}%")
    print("\n  Per-category keyword hit @ {}:".format(top_k))
    for cat in sorted(by_cat.keys()):
        lst = by_cat[cat]
        print(f"    {cat}: {sum(lst)}/{len(lst)} = {sum(lst)/len(lst)*100:.0f}%")
    print("=" * 70)

    if metrics_only:
        print("\nNote: Run without --metrics-only to see top-K per query.")
        return

    print(f"\nValidation queries (assignment categories) — Top {top_k} per query\n")
    print("=" * 70)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['category']}")
        print(f"    Query: \"{r['query']}\"")
        if r["expected_keywords"]:
            print(f"    Expected keywords: {r['expected_keywords']}  keyword_hit={r['hit']}")
        print(f"    Top {top_k}: {r['top_names']}")
    print("\n" + "=" * 70)
    print("\nNote: Hindi (Devanagari) — we support Romanized Hindi/Hinglish only.")
    print("      Catalog has vegetarian/non-vegetarian; no keto/jain tags.")
    print("      'Less oily' has no explicit metadata; may match via 'light' or veg.")


if __name__ == "__main__":
    main()
