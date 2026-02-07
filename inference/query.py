# Single-query ranking. Usage: python -m inference.query "query" [--lexical | --compare | --hybrid [w w w]] [--top N]

import csv
import re
import sys
from pathlib import Path

import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.train import load_checkpoint
from training.dataset import get_dish_display_text
from inference.bm25 import BM25


def _collapse_repeats(text: str) -> str:
    """Collapse repeated consecutive chars so 'daal' -> 'dal', 'fryy' -> 'fry'."""
    if not text:
        return text
    out = [text[0]]
    for c in text[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


def tokenize_for_lexical(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empty; minimal normalization."""
    text = (text or "").lower().strip()
    tokens = re.split(r"[^\w]+", text)
    return [t for t in tokens if len(t) >= 2]


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Used for fuzzy match (e.g. pattu vs puttu)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
        prev = curr
    return prev[m]


def lexical_score(query: str, dish_name: str, fuzzy: bool = True) -> float:
    """
    Score dish by query-term presence in dish name (no DL).
    - Exact/substring: query token in dish name (after collapsing repeats) → +1.0.
    - Fuzzy (if fuzzy=True): query token within edit distance 1 of any dish word → +0.85
      so e.g. "pattu" matches "puttu".
    """
    q_tokens = tokenize_for_lexical(query)
    name_lower = (dish_name or "").lower()
    name_norm = _collapse_repeats(name_lower)
    name_words = re.split(r"[^\w]+", name_lower)
    name_words = [w for w in name_words if len(w) >= 2]
    if not q_tokens:
        return 0.0
    score = 0.0
    for t in q_tokens:
        t_norm = _collapse_repeats(t)
        if t_norm in name_norm or t in name_lower:
            score += 1.0
        elif fuzzy and name_words:
            # One-edit fuzzy: e.g. pattu ↔ puttu
            for w in name_words:
                if abs(len(t) - len(w)) <= 1 and _edit_distance(t, w) <= 1:
                    score += 0.85
                    break
    match_len = sum(len(t) for t in q_tokens if _collapse_repeats(t) in name_norm or t in name_lower)
    return score + match_len * 0.001


# Terms that suggest user wants chaat-style dishes (bhalla, papdi, etc.)
_CHAAT_QUERY_TERMS = frozenset({
    "chaat", "bhalla", "papdi", "papri", "golgappa", "pani", "puri", "dahi", "vada",
    "sev", "bhel", "papadi", "chhole", "tikki", "samosa", "kachori",
})


def _query_for_scoring(query: str) -> str:
    """Append ' chaat' when query has chaat-style terms (bhalla, papdi, etc.) for better match."""
    q_lower = (query or "").lower()
    tokens = set(re.split(r"[^\w]+", q_lower)) & _CHAAT_QUERY_TERMS
    if tokens:
        return (query.strip() + " chaat").strip()
    return query.strip()


def rank_lexical(query: str, dish_names: list[str]) -> list[tuple[int, float]]:
    """Return list of (index, score) sorted by score descending."""
    scored = [(i, lexical_score(query, name)) for i, name in enumerate(dish_names)]
    scored.sort(key=lambda x: -x[1])
    return scored


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. Handles constant scores."""
    s_min = min(scores)
    s_max = max(scores)
    span = s_max - s_min
    if span <= 0:
        return [0.5] * len(scores)
    return [(s - s_min) / span for s in scores]


def hybrid_rank(
    query: str,
    dish_names: list[str],
    dish_display_texts: list[str],
    model=None,
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
    expand_chaat: bool = True,
) -> list[tuple[int, float]]:
    """Lexical + BM25 + DL, min-max norm each to [0,1], then weighted sum. Returns (idx, score) desc."""
    q = _query_for_scoring(query) if expand_chaat else query
    n = len(dish_names)
    # Lexical (query terms in dish name, typo-tolerant)
    lex_scores = [lexical_score(q, name) for name in dish_names]
    # BM25 over rich dish text (name | diet | course | flavor | region | state | style | ingredients)
    bm25 = BM25(dish_display_texts)
    bm25_scores = bm25.score(q)
    # DL
    if model is not None:
        with torch.no_grad():
            q_emb = model.encode([q])
            dish_embs = model.encode(dish_display_texts)
            dl_scores = (q_emb @ dish_embs.T).squeeze(0).tolist()
    else:
        dl_scores = [0.0] * n

    w_lex, w_bm25, w_dl = weights
    lex_n = _normalize_scores(lex_scores)
    bm25_n = _normalize_scores(bm25_scores)
    dl_n = _normalize_scores(dl_scores)
    combined = [
        w_lex * lex_n[i] + w_bm25 * bm25_n[i] + w_dl * dl_n[i]
        for i in range(n)
    ]
    ranked = sorted(enumerate(combined), key=lambda x: -x[1])
    return ranked


def main():
    args = sys.argv[1:]
    top_k = 10
    lexical_only = "--lexical" in args
    compare = "--compare" in args
    hybrid = "--hybrid" in args
    # Default hybrid weights: lexical, BM25, DL
    hybrid_weights: tuple[float, float, float] = (0.4, 0.35, 0.25)
    if "--hybrid" in args:
        i = args.index("--hybrid")
        # Optional: --hybrid 0.5 0.3 0.2
        rest = args[i + 1 : i + 4]
        if len(rest) >= 3:
            try:
                w = (float(rest[0]), float(rest[1]), float(rest[2]))
                if all(0 <= x <= 1 for x in w) and abs(sum(w) - 1.0) < 0.01:
                    hybrid_weights = w
                    args = args[:i] + args[i + 4 :]
                else:
                    args = args[:i] + args[i + 1 :]
            except ValueError:
                args = args[:i] + args[i + 1 :]
        else:
            args = args[:i] + args[i + 1 :]
    args = [a for a in args if a not in ("--lexical", "--compare")]
    if "--top" in args:
        i = args.index("--top")
        if i + 1 < len(args):
            top_k = int(args[i + 1])
        args = [a for j, a in enumerate(args) if a != "--top" and j != i + 1]
    query = " ".join(args).strip()
    if not query:
        print(
            "Usage: python -m inference.query \"query\" [--top N] [--lexical | --compare | --hybrid [w_lex w_bm25 w_dl]]"
        )
        sys.exit(1)

    dishes_csv = ROOT / "data" / "processed" / "dishes.csv"
    if not dishes_csv.exists():
        print(f"Dishes not found: {dishes_csv}")
        sys.exit(1)

    dishes = []
    with open(dishes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dishes.append(dict(row))
    dish_names = [d["name"] for d in dishes]
    dish_display_texts = [get_dish_display_text(d) for d in dishes]

    if hybrid:
        checkpoint_path = ROOT / "checkpoints" / "best_model.pt"
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        print(f"Loading model from {checkpoint_path}...")
        model, _ = load_checkpoint(str(checkpoint_path), device="cpu")
        ranked = hybrid_rank(
            query, dish_names, dish_display_texts, model=model, weights=hybrid_weights
        )
        print(
            f"\nQuery: \"{query}\" — HYBRID (lexical + BM25 + DL) weights={hybrid_weights}"
        )
        print(f"Top {top_k} dishes:\n")
        for r, (i, sc) in enumerate(ranked[:top_k], 1):
            print(f"  {r}. {dish_names[i]}  (score: {sc:.4f})")
        print()
        return

    if lexical_only or compare:
        lexical_ranked = rank_lexical(query, dish_names)
        print(f"\nQuery: \"{query}\" — LEXICAL baseline (query words in dish name)")
        print(f"Top {top_k} dishes:\n")
        for r, (i, sc) in enumerate(lexical_ranked[:top_k], 1):
            print(f"  {r}. {dish_names[i]}  (score: {sc:.4f})")
        if compare:
            print()

    if not lexical_only:
        checkpoint_path = ROOT / "checkpoints" / "best_model.pt"
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        print(f"Loading model from {checkpoint_path}...")
        model, _ = load_checkpoint(str(checkpoint_path), device="cpu")
        with torch.no_grad():
            q_emb = model.encode([query])
            dish_embs = model.encode(dish_display_texts)
            scores = (q_emb @ dish_embs.T).squeeze(0)
        ranked_idx = scores.argsort(descending=True)
        print(f"\nQuery: \"{query}\" — DL model (best_model.pt)")
        print(f"Top {top_k} dishes:\n")
        for r, i in enumerate(ranked_idx[:top_k], 1):
            print(f"  {r}. {dish_names[i]}  (score: {scores[i].item():.4f})")
    print()


if __name__ == "__main__":
    main()
