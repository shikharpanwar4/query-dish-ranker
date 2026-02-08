# Eval: bi-encoder only and hybrid (lexical + BM25 + bi-encoder). R@1, R@5, R@10, MRR on val set.
# Usage: python -m inference.eval_hybrid [--mode bi|hybrid] [--weights w w w] [--limit N] [--per-type] [--tune-weights]

import csv
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    print("Missing dependency: torch. pip install -r requirements.txt")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.train import load_checkpoint, infer_query_type
from training.dataset import get_dish_display_text
from inference.query import hybrid_rank
from inference.bm25 import BM25

K_VALUES = [1, 5, 10, 50]


def _compute_metrics(ranked_indices_by_query, valid_pairs, dish_names, k_values):
    """ranked_indices_by_query: list of list of (idx, score) or just list of indices per query, sorted desc."""
    correct_at_k = {k: 0 for k in k_values}
    rr_sum = 0.0
    n = len(valid_pairs)
    dish_to_idx = {name: i for i, name in enumerate(dish_names)}
    for (query, correct_dish), ranked in zip(valid_pairs, ranked_indices_by_query):
        if isinstance(ranked[0], tuple):
            indices = [idx for idx, _ in ranked]
        else:
            indices = ranked
        rank = None
        correct_idx = dish_to_idx.get(correct_dish)
        if correct_idx is None:
            continue
        for r, idx in enumerate(indices):
            if idx == correct_idx:
                rank = r
                break
        if rank is None:
            continue
        for k in k_values:
            if rank < k:
                correct_at_k[k] += 1
        rr_sum += 1.0 / (rank + 1)
    return {
        **{f"R@{k}": correct_at_k[k] / max(n, 1) for k in k_values},
        "MRR": rr_sum / max(n, 1),
        "n": n,
    }


def eval_bi_only(model, valid_pairs, dish_names, dish_display_texts, device="cpu"):
    """Rank full catalog with bi-encoder (DL) only; no lexical/BM25."""
    with torch.no_grad():
        dish_embs = model.encode(dish_display_texts)
    ranked_per_query = []
    for query, _ in valid_pairs:
        with torch.no_grad():
            q_emb = model.encode([query])
            scores = (q_emb @ dish_embs.T).squeeze(0)
        idx_scores = sorted(enumerate(scores.tolist()), key=lambda x: -x[1])
        ranked_per_query.append(idx_scores)
    return _compute_metrics(ranked_per_query, valid_pairs, dish_names, K_VALUES)


def eval_hybrid_only(model, valid_pairs, dish_names, dish_display_texts, weights, precomputed_dish_embs=None, bm25=None):
    """Stage 1 only: lexical + BM25 + bi-encoder. Pass precomputed_dish_embs/bm25 for fast eval."""
    ranked_per_query = []
    for query, _ in valid_pairs:
        ranked = hybrid_rank(
            query, dish_names, dish_display_texts, model=model, weights=weights,
            precomputed_dish_embs=precomputed_dish_embs, bm25=bm25,
        )
        ranked_per_query.append(ranked)
    return _compute_metrics(ranked_per_query, valid_pairs, dish_names, K_VALUES)


def eval_hybrid_per_type(model, valid_pairs, dish_names, dish_display_texts, weights, precomputed_dish_embs=None, bm25=None):
    """Per-query-type breakdown of hybrid ranking. Returns {query_type: metrics_dict}."""
    from collections import defaultdict
    dish_to_idx = {name: i for i, name in enumerate(dish_names)}
    type_ranked = defaultdict(list)  # type -> [(pair, ranked)]
    for query, correct_dish in valid_pairs:
        ranked = hybrid_rank(
            query, dish_names, dish_display_texts, model=model, weights=weights,
            precomputed_dish_embs=precomputed_dish_embs, bm25=bm25,
        )
        qtype = infer_query_type(query, correct_dish)
        type_ranked[qtype].append(((query, correct_dish), ranked))
    results = {}
    for qtype, items in sorted(type_ranked.items()):
        pairs = [pair for pair, _ in items]
        rankings = [ranked for _, ranked in items]
        m = _compute_metrics(rankings, pairs, dish_names, K_VALUES)
        results[qtype] = m
    return results


def _precompute_all_scores(model, valid_pairs, dish_names, dish_display_texts, precomputed_dish_embs, bm25):
    """Precompute lexical, BM25, and DL scores for all (query, dish) combos.

    Returns three numpy arrays of shape [num_queries, num_dishes],
    each min-max normalized to [0, 1].
    """
    import numpy as np
    from inference.query import lexical_score, _query_for_scoring, _normalize_scores
    n_queries = len(valid_pairs)
    n_dishes = len(dish_names)
    lex_all = np.zeros((n_queries, n_dishes), dtype=np.float32)
    bm25_all = np.zeros((n_queries, n_dishes), dtype=np.float32)
    dl_all = np.zeros((n_queries, n_dishes), dtype=np.float32)
    print(f"  Precomputing scores for {n_queries} queries × {n_dishes} dishes...")
    for qi, (query, _) in enumerate(valid_pairs):
        q = _query_for_scoring(query)
        # Lexical
        lex_raw = [lexical_score(q, name) for name in dish_names]
        lex_all[qi] = _normalize_scores(lex_raw)
        # BM25
        bm25_raw = bm25.score(q)
        bm25_all[qi] = _normalize_scores(bm25_raw)
        # DL
        with torch.no_grad():
            q_emb = model.encode([q])
            dl_raw = (q_emb @ precomputed_dish_embs.T).squeeze(0).tolist()
        dl_all[qi] = _normalize_scores(dl_raw)
        if (qi + 1) % 100 == 0:
            print(f"    {qi + 1}/{n_queries} queries precomputed")
    return lex_all, bm25_all, dl_all


def tune_weights(model, valid_pairs, dish_names, dish_display_texts, precomputed_dish_embs, bm25, metric="MRR", step=0.05):
    """Grid search over hybrid weights using precomputed scores. Very fast after precompute.

    Returns (best_weights, best_metric, all_results).
    """
    import numpy as np
    # Phase 1: precompute all scores (this is the slow part, but done only once)
    lex_all, bm25_all, dl_all = _precompute_all_scores(
        model, valid_pairs, dish_names, dish_display_texts, precomputed_dish_embs, bm25,
    )
    # Build dish_to_idx for fast lookup
    dish_to_idx = {name: i for i, name in enumerate(dish_names)}
    correct_indices = []
    valid_mask = []
    for query, correct_dish in valid_pairs:
        idx = dish_to_idx.get(correct_dish)
        if idx is not None:
            correct_indices.append(idx)
            valid_mask.append(True)
        else:
            correct_indices.append(0)
            valid_mask.append(False)
    correct_indices = np.array(correct_indices)
    valid_mask = np.array(valid_mask)
    n_valid = valid_mask.sum()

    # Phase 2: fast grid search (just numpy weighted sums + rank computation)
    steps_arr = np.arange(0.0, 1.01, step)
    combos = []
    for w_lex in steps_arr:
        for w_bm25 in steps_arr:
            w_dl = 1.0 - w_lex - w_bm25
            if w_dl < -0.01 or w_dl > 1.01:
                continue
            combos.append((round(float(w_lex), 3), round(float(w_bm25), 3), round(max(0.0, float(w_dl)), 3)))
    print(f"  Grid search: {len(combos)} combos (step={step}), metric={metric}")

    best_weights = (0.4, 0.35, 0.25)
    best_score = -1.0
    all_results = []
    for ci, (w_lex, w_bm25, w_dl) in enumerate(combos):
        combined = w_lex * lex_all + w_bm25 * bm25_all + w_dl * dl_all  # [Q, D]
        # Compute rank of correct dish per query using argsort (handles ties correctly)
        rr_sum = 0.0
        correct_at_k = {}
        for qi in range(len(valid_pairs)):
            if not valid_mask[qi]:
                continue
            # argsort descending: rank = position of correct dish in sorted order
            order = (-combined[qi]).argsort()
            rank = int(np.where(order == correct_indices[qi])[0][0])
            rr_sum += 1.0 / (rank + 1)
            for kk in [1, 5, 10, 50]:
                correct_at_k[kk] = correct_at_k.get(kk, 0) + (1 if rank < kk else 0)
        if metric == "MRR":
            score = rr_sum / max(n_valid, 1)
        else:
            k = int(metric.replace("R@", ""))
            score = correct_at_k.get(k, 0) / max(n_valid, 1)
        all_results.append(((w_lex, w_bm25, w_dl), score))
        if score > best_score:
            best_score = score
            best_weights = (w_lex, w_bm25, w_dl)
        if (ci + 1) % 50 == 0:
            print(f"    {ci + 1}/{len(combos)} ... current best: {best_weights} {metric}={best_score:.4f}")
    return best_weights, best_score, all_results


def main():
    data_dir = ROOT / "data" / "processed"
    val_csv = data_dir / "val.csv"
    dishes_csv = data_dir / "dishes.csv"
    checkpoint_path = ROOT / "checkpoints" / "best_model.pt"
    for p in (val_csv, dishes_csv, checkpoint_path):
        if not p.exists():
            print(f"Missing: {p}")
            sys.exit(1)

    # Load dishes
    dishes = []
    with open(dishes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dishes.append(dict(row))
    dish_names = [d["name"] for d in dishes]
    dish_display_texts = [get_dish_display_text(d) for d in dishes]

    # Load val pairs
    val_pairs = []
    with open(val_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("query") or "").strip()
            d = (row.get("dish_name") or row.get("dish") or "").strip()
            if q and d:
                val_pairs.append((q, d))
    dish_to_idx = {name: i for i, name in enumerate(dish_names)}
    valid_pairs = [(q, d) for q, d in val_pairs if d in dish_to_idx]
    if len(valid_pairs) < len(val_pairs):
        print(f"Skipped {len(val_pairs) - len(valid_pairs)} val pairs (dish not in catalog)")

    limit = None
    if "--limit" in sys.argv:
        i = sys.argv.index("--limit")
        if i + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[i + 1])
                valid_pairs = valid_pairs[:limit]
                print(f"Limited to first {limit} val pairs.")
            except ValueError:
                pass

    weights = (0.10, 0.45, 0.45)
    if "--weights" in sys.argv:
        i = sys.argv.index("--weights")
        if i + 3 <= len(sys.argv):
            try:
                weights = (float(sys.argv[i + 1]), float(sys.argv[i + 2]), float(sys.argv[i + 3]))
            except ValueError:
                pass

    mode = "all"
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        if i + 1 < len(sys.argv):
            mode = sys.argv[i + 1].lower()

    device = "cpu"
    print(f"Loading bi-encoder from {checkpoint_path}...")
    model, _ = load_checkpoint(str(checkpoint_path), device=device)
    n = len(valid_pairs)
    print(f"Val pairs: {n}  |  Catalog: {len(dish_names)}  |  Weights: {weights}")

    # Precompute once for fast eval
    with torch.no_grad():
        precomputed_dish_embs = model.encode(dish_display_texts)
    bm25_index = BM25(dish_display_texts)
    print("Precomputed dish embeddings and BM25 index.")
    print()

    def print_metrics(name, m):
        print(f"  {name}")
        for k in K_VALUES:
            print(f"    R@{k}:  {m[f'R@{k}']:.4f}")
        print(f"    MRR:  {m['MRR']:.4f}")
        print()

    results = []

    if mode in ("all", "bi"):
        m_bi = eval_bi_only(model, valid_pairs, dish_names, dish_display_texts, device)
        results.append(("Bi-encoder only", "1. Bi-encoder only (DL vs full catalog)", m_bi))

    if mode in ("all", "hybrid"):
        m_hybrid = eval_hybrid_only(
            model, valid_pairs, dish_names, dish_display_texts, weights,
            precomputed_dish_embs=precomputed_dish_embs, bm25=bm25_index,
        )
        results.append(("Hybrid", "2. Hybrid (lexical + BM25 + bi-encoder)", m_hybrid))

    for name, title, m in results:
        print("=" * 50)
        print(title)
        print("=" * 50)
        print_metrics("", m)

    if mode == "all" and results:
        print("=" * 50)
        print("Summary")
        print("=" * 50)
        for name, _, m in results:
            print(f"  {name}: R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} R@10={m['R@10']:.4f} MRR={m['MRR']:.4f}")
        print("=" * 50)

    # --- Per-type breakdown ---
    if "--per-type" in sys.argv:
        print()
        print("=" * 72)
        print("Per-query-type breakdown (hybrid)")
        print("=" * 72)
        pt = eval_hybrid_per_type(
            model, valid_pairs, dish_names, dish_display_texts, weights,
            precomputed_dish_embs=precomputed_dish_embs, bm25=bm25_index,
        )
        print(f"  {'Type':<12s} {'Count':>6s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s} {'R@50':>7s} {'MRR':>7s}")
        print("  " + "-" * 58)
        for qtype in sorted(pt.keys(), key=lambda t: -pt[t]["n"]):
            m = pt[qtype]
            print(f"  {qtype:<12s} {m['n']:>6d} {m['R@1']:>7.3f} {m['R@5']:>7.3f} {m['R@10']:>7.3f} {m['R@50']:>7.3f} {m['MRR']:>7.3f}")
        total = sum(m["n"] for m in pt.values())
        print("  " + "-" * 58)
        print(f"  {'TOTAL':<12s} {total:>6d}")
        print("=" * 72)

    # --- Weight tuning ---
    if "--tune-weights" in sys.argv:
        print()
        tune_step = 0.05
        tune_metric = "MRR"
        if "--tune-metric" in sys.argv:
            i = sys.argv.index("--tune-metric")
            if i + 1 < len(sys.argv):
                tune_metric = sys.argv[i + 1]
        best_w, best_s, all_res = tune_weights(
            model, valid_pairs, dish_names, dish_display_texts,
            precomputed_dish_embs, bm25_index, metric=tune_metric, step=tune_step,
        )
        print()
        print("=" * 50)
        print(f"Best weights ({tune_metric}={best_s:.4f}): lexical={best_w[0]}, BM25={best_w[1]}, DL={best_w[2]}")
        print("=" * 50)
        # Show metrics with best weights
        m_best = eval_hybrid_only(
            model, valid_pairs, dish_names, dish_display_texts, best_w,
            precomputed_dish_embs=precomputed_dish_embs, bm25=bm25_index,
        )
        print_metrics("  Best weights", m_best)
        # Top 10 combos
        all_res.sort(key=lambda x: -x[1])
        print("  Top 10 weight combos:")
        for w, s in all_res[:10]:
            print(f"    ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}) → {tune_metric}={s:.4f}")


if __name__ == "__main__":
    main()
