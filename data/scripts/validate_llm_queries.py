#!/usr/bin/env python3
"""Validate llm_queries*.json format and report how many resolve to the catalog. Usage: python data/scripts/validate_llm_queries.py [path/to/llm_queries.json] [path/to/dishes.csv]."""

import json
import sys
from collections import defaultdict
from pathlib import Path

STOP_WORDS = {"and", "with", "the", "of", "in", "a", "or", "for", "to", "on", "at", "by", "&", "-", "|"}


def build_keyword_to_dishes_from_csv(dishes_path: Path) -> tuple[dict[str, list[str]], set[str]]:
    """Build keyword -> dish names from dishes.csv; return (index, valid_dish_names_lower)."""
    import csv
    index = defaultdict(list)
    valid = set()
    with open(dishes_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            valid.add(name.lower())
            name_lower = name.lower()
            index[name_lower].append(name)
            for w in name.split():
                w = w.lower()
                if len(w) >= 2 and w not in STOP_WORDS:
                    if name not in index[w]:
                        index[w].append(name)
            for col in ("ingredients", "diet", "flavor", "course", "state", "region", "category"):
                val = (row.get(col) or "").strip().lower()
                if val:
                    for w in val.replace(",", " ").split():
                        w = w.strip()
                        if len(w) >= 2 and w not in STOP_WORDS and name not in index.get(w, []):
                            index[w].append(name)
    return dict(index), valid


def resolve_keywords(expected_keywords: list, keyword_to_dishes: dict, valid: set) -> str | None:
    from collections import Counter
    candidates = []
    for kw in expected_keywords or []:
        k = (kw or "").strip().lower()
        if not k:
            continue
        for dish in keyword_to_dishes.get(k, []):
            if dish.lower() in valid:
                candidates.append(dish)
    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]


def main():
    args = sys.argv[1:]
    json_path = Path(args[0]) if args else Path(__file__).resolve().parent.parent / "raw" / "llm_queries.json"
    dishes_path = Path(args[1]) if len(args) >= 2 else Path(__file__).resolve().parent.parent / "processed" / "dishes.csv"

    if not json_path.exists():
        print(f"JSON not found: {json_path}")
        print("Usage: python validate_llm_queries.py [llm_queries.json] [dishes.csv]")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Error: root must be a JSON array")
        sys.exit(1)

    by_category = defaultdict(int)
    with_dish = 0
    with_keywords = 0
    bad = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            bad += 1
            continue
        q = (item.get("query") or "").strip()
        if not q:
            bad += 1
            continue
        if item.get("dish"):
            with_dish += 1
        if item.get("expected_keywords") is not None:
            with_keywords += 1
        cat = (item.get("category") or "").strip()
        if cat:
            by_category[cat] += 1

    print(f"Total items: {len(data)}")
    print(f"With 'dish': {with_dish} | With 'expected_keywords': {with_keywords} | Bad/missing query: {bad}")
    print("By category:", dict(by_category))

    if dishes_path.exists() and (with_keywords or with_dish):
        keyword_to_dishes, valid = build_keyword_to_dishes_from_csv(dishes_path)
        resolved = 0
        unresolved = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            query = (item.get("query") or "").strip()
            if not query:
                continue
            if item.get("dish"):
                dish = (item.get("dish") or "").strip()
                if dish.lower() in valid:
                    resolved += 1
                else:
                    unresolved += 1
            elif item.get("expected_keywords") is not None:
                kw_list = item["expected_keywords"] if isinstance(item["expected_keywords"], list) else [item["expected_keywords"]]
                if resolve_keywords(kw_list, keyword_to_dishes, valid):
                    resolved += 1
                else:
                    unresolved += 1
        print(f"Catalog resolution: {resolved} resolved, {unresolved} unresolved (catalog: {dishes_path})")
    else:
        if not dishes_path.exists():
            print(f"Optional: provide dishes.csv to check resolution (e.g. {dishes_path})")

    print("OK: valid JSON and structure.")


if __name__ == "__main__":
    main()
