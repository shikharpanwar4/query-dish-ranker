# End-to-end latency: query in → top-K out (500 dishes, CPU). Breakdown for optimization. Target: <100 ms.

import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.train import load_checkpoint
from training.dataset import get_dish_display_text
from inference.query import (
    _query_for_scoring,
    lexical_score,
    _normalize_scores,
)
from inference.bm25 import BM25


def main():
    data_dir = ROOT / "data" / "processed"
    dishes_csv = data_dir / "dishes.csv"
    checkpoint_path = ROOT / "checkpoints" / "best_model.pt"

    if not dishes_csv.exists() or not checkpoint_path.exists():
        print("Run prepare_data and training first.")
        sys.exit(1)

    dishes = []
    with open(dishes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dishes.append(dict(row))

    n_dishes = 500
    if len(dishes) < n_dishes:
        while len(dishes) < n_dishes:
            dishes.extend(dishes[: n_dishes - len(dishes)])
    dishes = dishes[:n_dishes]
    dish_names = [d["name"] for d in dishes]
    dish_display_texts = [get_dish_display_text(d) for d in dishes]

    print("Loading model...")
    model, _ = load_checkpoint(str(checkpoint_path), device="cpu")
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 * 1024)
    print(f"  Params: {total_params:,}  (~{size_mb:.2f} MB)")
    print(f"  Requirement: <20 MB  →  {'PASS' if size_mb < 20 else 'FAIL'}")
    print()

    print("Precomputing dish embeddings and BM25 index...")
    with torch.no_grad():
        dish_embs = model.encode(dish_display_texts)
    bm25 = BM25(dish_display_texts)
    weights = (0.4, 0.35, 0.25)
    n = len(dish_names)
    print("Done.")
    print()

    test_queries = [
        "butter chicken", "spicy paneer", "dal fry", "chaat", "south indian breakfast",
        "veg dessert", "meetha kuch", "paneer tikka", "biryani", "dosa",
        "bhalla papadi", "masaledar khana", "chhole kulche", "pattu", "dal fryy",
    ]
    while len(test_queries) < 20:
        test_queries.append(test_queries[-1])

    # Warmup
    for _ in range(3):
        q = _query_for_scoring(test_queries[0])
        _ = [lexical_score(q, name) for name in dish_names]
        _ = bm25.score(q)
        with torch.no_grad():
            _ = model.encode([q]) @ dish_embs.T

    # Per-stage timings (same order as hot path)
    t_query_norm = []
    t_lexical = []
    t_bm25 = []
    t_dl = []
    t_combine_sort = []
    e2e_ms = []

    for query in test_queries:
        # Query norm
        t0 = time.perf_counter()
        q = _query_for_scoring(query)
        t1 = time.perf_counter()
        t_query_norm.append((t1 - t0) * 1000)

        # Lexical (500 dishes)
        t0 = time.perf_counter()
        lex_scores = [lexical_score(q, name) for name in dish_names]
        t1 = time.perf_counter()
        t_lexical.append((t1 - t0) * 1000)

        # BM25
        t0 = time.perf_counter()
        bm25_scores = bm25.score(q)
        t1 = time.perf_counter()
        t_bm25.append((t1 - t0) * 1000)

        # DL: encode query + dot with precomputed dish_embs
        t0 = time.perf_counter()
        with torch.no_grad():
            q_emb = model.encode([q])
            dl_scores = (q_emb @ dish_embs.T).squeeze(0).tolist()
        t1 = time.perf_counter()
        t_dl.append((t1 - t0) * 1000)

        # Normalize + combine + sort
        t0 = time.perf_counter()
        lex_n = _normalize_scores(lex_scores)
        bm25_n = _normalize_scores(bm25_scores)
        dl_n = _normalize_scores(dl_scores)
        combined = [
            weights[0] * lex_n[i] + weights[1] * bm25_n[i] + weights[2] * dl_n[i]
            for i in range(n)
        ]
        sorted(enumerate(combined), key=lambda x: -x[1])
        t1 = time.perf_counter()
        t_combine_sort.append((t1 - t0) * 1000)

        e2e_ms.append(t_query_norm[-1] + t_lexical[-1] + t_bm25[-1] + t_dl[-1] + t_combine_sort[-1])

    def mean_ms(x):
        return sum(x) / len(x) if x else 0

    e2e_mean = mean_ms(e2e_ms)
    e2e_p50 = sorted(e2e_ms)[len(e2e_ms) // 2]
    e2e_p99 = sorted(e2e_ms)[int(len(e2e_ms) * 0.99)] if len(e2e_ms) >= 10 else max(e2e_ms)

    m_q = mean_ms(t_query_norm)
    m_lex = mean_ms(t_lexical)
    m_bm25 = mean_ms(t_bm25)
    m_dl = mean_ms(t_dl)
    m_comb = mean_ms(t_combine_sort)

    print("End-to-end latency: 1 query → ranked top-K (500 dishes, CPU, precomputed dish embs)")
    print("=" * 65)
    print(f"  End-to-end mean:  {e2e_mean:.2f} ms")
    print(f"  End-to-end P50:  {e2e_p50:.2f} ms")
    print(f"  End-to-end P99:  {e2e_p99:.2f} ms")
    print(f"  Requirement: <100 ms (CPU)  →  {'PASS' if e2e_mean < 100 else 'FAIL'}")
    print()
    print("Breakdown (mean ms per query, % of end-to-end):")
    print("-" * 65)
    for label, ms in [
        ("  query norm (chaat expand)", m_q),
        ("  lexical (500 × fuzzy match)", m_lex),
        ("  BM25 (500 docs)", m_bm25),
        ("  DL (encode query + dot)", m_dl),
        ("  normalize + combine + sort", m_comb),
    ]:
        pct = 100 * ms / e2e_mean if e2e_mean > 0 else 0
        print(f"  {label:<32}  {ms:6.2f} ms  ({pct:5.1f}%)")
    print("-" * 65)
    print(f"  {'(sum)':<32}  {m_q + m_lex + m_bm25 + m_dl + m_comb:6.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
