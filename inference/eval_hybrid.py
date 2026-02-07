# Hybrid ranker on val set: R@1, R@5, R@10, MRR. Usage: python -m inference.eval_hybrid [--weights w w w]

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.train import load_checkpoint
from training.dataset import get_dish_display_text
from inference.query import hybrid_rank


def main():
    data_dir = ROOT / "data" / "processed"
    val_csv = data_dir / "val.csv"
    dishes_csv = data_dir / "dishes.csv"
    checkpoint_path = ROOT / "checkpoints" / "best_model.pt"

    for p in (val_csv, dishes_csv, checkpoint_path):
        if not p.exists():
            print(f"Missing: {p}")
            sys.exit(1)

    # Load dishes and rich display texts
    dishes = []
    with open(dishes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dishes.append(dict(row))
    dish_names = [d["name"] for d in dishes]
    dish_display_texts = [get_dish_display_text(d) for d in dishes]

    # Load val pairs (query, correct_dish_name)
    val_pairs = []
    with open(val_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("query") or "").strip()
            d = (row.get("dish_name") or row.get("dish") or "").strip()
            if q and d:
                val_pairs.append((q, d))

    dish_to_idx = {name: i for i, name in enumerate(dish_names)}
    valid_pairs = [(q, d) for q, d in val_pairs if d in dish_to_idx]
    skipped = len(val_pairs) - len(valid_pairs)
    if skipped:
        print(f"Skipped {skipped} val pairs (dish not in catalog)")

    print(f"Loading model from {checkpoint_path}...")
    model, _ = load_checkpoint(str(checkpoint_path), device="cpu")

    # Default hybrid weights
    weights = (0.4, 0.35, 0.25)
    if "--weights" in sys.argv:
        i = sys.argv.index("--weights")
        if i + 3 <= len(sys.argv):
            try:
                weights = (float(sys.argv[i + 1]), float(sys.argv[i + 2]), float(sys.argv[i + 3]))
            except ValueError:
                pass
    print(f"Hybrid weights: {weights}")

    k_values = [1, 5, 10]
    correct_at_k = {k: 0 for k in k_values}
    rr_sum = 0.0
    n = len(valid_pairs)

    for query, correct_dish in valid_pairs:
        ranked = hybrid_rank(
            query, dish_names, dish_display_texts, model=model, weights=weights
        )
        # ranked = [(idx, score), ...] sorted by score desc
        rank = None
        for r, (idx, _) in enumerate(ranked):
            if dish_names[idx] == correct_dish:
                rank = r  # 0-indexed
                break
        if rank is None:
            continue
        for k in k_values:
            if rank < k:
                correct_at_k[k] += 1
        rr_sum += 1.0 / (rank + 1)

    print("\n" + "=" * 50)
    print("Hybrid ranker evaluation (val set)")
    print("=" * 50)
    print(f"  Valid pairs:  {n}")
    print(f"  Catalog:     {len(dish_names)} dishes")
    print(f"  Weights:     lex={weights[0]}, bm25={weights[1]}, dl={weights[2]}")
    print()
    for k in k_values:
        rk = correct_at_k[k] / max(n, 1)
        print(f"  R@{k}:  {rk:.4f}")
    print(f"  MRR:   {rr_sum / max(n, 1):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
