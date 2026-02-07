"""
Dual-encoder scorer: scores relevance between a query and a dish name.

ARCHITECTURE:
  The scorer wraps a SINGLE shared TextEncoder and uses it for both queries
  and dish names. This is called a "Siamese" or "dual encoder" architecture.

  WHY SHARED WEIGHTS:
    Queries and dishes live in the same linguistic space — "paneer tikka"
    could be either a query or a dish name. Sharing weights means:
    1. Half the parameters (6.4M instead of 12.8M) — fits our budget
    2. Built-in regularization — the encoder must learn representations
       that work for both input types
    3. Symmetric similarity — score(A, B) == score(B, A), which makes
       sense for text relevance

  WHY NOT SEPARATE ENCODERS:
    Separate encoders are useful when inputs are structurally different
    (e.g., images vs text in CLIP). Our inputs are both short text,
    so separate encoders would just overfit.

PRODUCTION USAGE:
  The key insight for production: dish embeddings are PRECOMPUTED.

  Offline (once):
    dish_embeddings = scorer.encode_dishes(all_dish_names)  # [N, 128]
    save(dish_embeddings)

  Online (per query, <2ms):
    q_emb = scorer.encode_query(query)           # [1, 128]  ~1ms
    scores = q_emb @ dish_embeddings.T           # [1, N]    ~0.05ms
    top_k = scores.topk(k)                       # ~0.01ms
"""

import torch
import torch.nn as nn

from model.tokenizer import CharTrigramTokenizer
from model.encoder import TextEncoder


class DualEncoderScorer(nn.Module):
    """
    End-to-end scorer: raw text → relevance score.

    Wraps tokenizer + encoder into a single module for clean APIs.
    The tokenizer is not an nn.Module (no parameters), but the encoder is.
    """

    def __init__(
        self,
        num_buckets: int = 25_000,
        embed_dim: int = 80,
        dropout: float = 0.18,
        max_trigrams: int = 48,
    ):
        super().__init__()

        # Tokenizer is stateless — not a nn.Module, no parameters
        self.tokenizer = CharTrigramTokenizer(
            num_buckets=num_buckets,
            max_trigrams=max_trigrams,
        )

        # Single shared encoder for both queries and dishes
        self.encoder = TextEncoder(
            num_buckets=num_buckets,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.embed_dim = embed_dim

    def _tokenize(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenize a list of texts and return as tensor.

        Note: we move the tensor to the same device as the model.
        This is important for GPU training — tokenization happens on
        CPU (it's just hashing), but the tensor must move to GPU
        before entering the encoder.
        """
        ids = self.tokenizer.encode_batch(texts)
        device = next(self.encoder.parameters()).device
        return torch.tensor(ids, dtype=torch.long, device=device)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        This is the core method — used for both queries and dishes
        (same encoder, same method). The "dual" in dual-encoder refers
        to the two inputs being encoded independently, not two models.

        Args:
            texts: list of N strings

        Returns:
            [N, embed_dim] — L2-normalized embeddings
        """
        token_ids = self._tokenize(texts)
        return self.encoder(token_ids)

    def forward(
        self,
        query_ids: torch.Tensor,
        dish_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between query and dish batches.

        Used during TRAINING where we already have tokenized IDs
        (from the DataLoader) and need the full similarity matrix
        for the InfoNCE loss.

        Args:
            query_ids: [batch, max_trigrams] — tokenized queries
            dish_ids:  [batch, max_trigrams] — tokenized dishes

        Returns:
            [batch, batch] — cosine similarity matrix
                             sim[i][j] = similarity of query_i with dish_j
                             Diagonal = positive pairs
        """
        query_embs = self.encoder(query_ids)  # [batch, embed_dim]
        dish_embs = self.encoder(dish_ids)    # [batch, embed_dim]

        # Matrix multiply: each query against each dish
        # Since embeddings are L2-normalized, this IS cosine similarity
        similarity = query_embs @ dish_embs.T  # [batch, batch]

        return similarity

    def score_pairs(
        self,
        queries: list[str],
        dishes: list[str],
    ) -> list[float]:
        """
        Score a list of (query, dish) pairs.

        Convenience method for evaluation / demo. Not used in training.

        Args:
            queries: list of N query strings
            dishes:  list of N dish strings

        Returns:
            list of N similarity scores in [-1, 1]
        """
        query_embs = self.encode(queries)  # [N, embed_dim]
        dish_embs = self.encode(dishes)    # [N, embed_dim]

        # Element-wise dot product (not matrix multiply — just paired scores)
        scores = (query_embs * dish_embs).sum(dim=-1)  # [N]
        return scores.tolist()

    def score_one_to_many(
        self,
        query: str,
        dishes: list[str],
    ) -> list[float]:
        """
        Score one query against many dishes.

        This is the PRODUCTION inference pattern:
          1. Encode query (1 forward pass)
          2. Encode all dishes (or load pre-computed)
          3. Matrix multiply → all scores at once

        Args:
            query:  single query string
            dishes: list of M dish strings

        Returns:
            list of M similarity scores
        """
        query_emb = self.encode([query])    # [1, embed_dim]
        dish_embs = self.encode(dishes)     # [M, embed_dim]

        scores = (query_emb @ dish_embs.T).squeeze(0)  # [M]
        return scores.tolist()


# ---------------------------------------------------------------------------
# Quick self-test — run with: python -m model.scorer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = DualEncoderScorer()

    # Count parameters
    total = sum(p.numel() for p in scorer.parameters())
    print(f"Total parameters: {total:,}")
    size_mb = total * 4 / 1024 / 1024
    print(f"Model size: {size_mb:.1f} MB (float32), {size_mb/2:.1f} MB (float16)")
    print()

    # --- Test: score pairs ---
    print("=== Pair scoring (untrained) ===")
    queries = ["spicy paneer", "meetha chahiye", "chicken biryani"]
    dishes  = ["Shahi paneer",  "Gulab jamun",    "Chicken Biryani"]

    scorer.eval()
    with torch.no_grad():
        scores = scorer.score_pairs(queries, dishes)
    for q, d, s in zip(queries, dishes, scores):
        print(f"  Q: {q:<25s} D: {d:<25s} Score: {s:.3f}")

    # --- Test: one-to-many ---
    print("\n=== One query vs many dishes (untrained) ===")
    query = "butter chicken"
    all_dishes = [
        "Butter chicken", "Paneer butter masala", "Dal makhani",
        "Gulab jamun", "Idli", "Samosa",
    ]
    with torch.no_grad():
        scores = scorer.score_one_to_many(query, all_dishes)

    ranked = sorted(zip(all_dishes, scores), key=lambda x: -x[1])
    print(f"  Query: '{query}'")
    for dish, score in ranked:
        print(f"    {score:+.3f}  {dish}")

    print(f"\nNote: rankings are random — model is untrained.")
    print(f"After training, 'Butter chicken' should rank first.")
