"""
Text encoder: trigram IDs → dense embedding vector.

Architecture (Deep Averaging Network — balanced variant):
  1. Embedding lookup: each trigram ID → 80d vector
  2. Mean pooling (masked): average non-padding vectors → single 80d
  3. Projection: LayerNorm → Linear(80→80) → GELU → Dropout
  4. L2 normalize: project onto unit sphere for cosine similarity

WHY THIS SIZE (Option B):
  With ~91K training pairs and 6,586 dishes, we can afford more capacity
  than the earlier 965K-param slim model, but still stay well under 10M:
    - 6,586 dishes need ~13 dimensions to separate (log2(6586))
    - 91K training pairs / 1.6M params = 0.057 ratio (healthy)
    - 80d embeddings give 25% more capacity than 64d for disambiguating
      10x more dishes, while staying CPU-friendly
    - 25K hash buckets for current data (~15–25K distinct trigrams)

PARAMETER BUDGET:
  Embedding:  25,000 × 80  = 2,000,000
  LayerNorm:  80 × 2       =       160
  Linear:     80 × 80 + 80 =     6,480
  Total:                     ≈   2.0M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    Encodes a batch of trigram ID sequences into L2-normalized embeddings.

    Input:  [batch_size, max_trigrams]  — integer tensor of hash IDs
    Output: [batch_size, embed_dim]     — unit-length embedding vectors
    """

    def __init__(
        self,
        num_buckets: int = 25_000,
        embed_dim: int = 80,
        dropout: float = 0.18,
        pad_id: int = 0,
    ):
        """
        Args:
            num_buckets: Hash space size. 25K for current data; lower collisions.
            embed_dim:   80d gives enough capacity to disambiguate 6.4K dishes.
            dropout:     0.18 for better generalization (reduce overfitting).
            pad_id:      Bucket 0 reserved for padding (zero vector).
        """
        super().__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim

        # --- Embedding table ---
        self.embedding = nn.Embedding(
            num_embeddings=num_buckets,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        # --- Single-layer projection ---
        # One linear layer with GELU is enough for our task.
        # The nonlinearity lets the model learn that "butter" + "chicken"
        # maps differently than each word alone. Without GELU, two
        # linear operations (embedding avg + linear) collapse to one.
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear layers with Xavier uniform.

        Why Xavier (not Kaiming)?
          - Kaiming is designed for ReLU, which kills half the distribution.
          - We use GELU, which is closer to linear around zero.
          - Xavier assumes symmetric activation, which fits GELU better.

        We also initialize the embedding table with a small normal distribution.
        The default PyTorch init (N(0,1)) gives huge initial vectors that
        dominate the first few training steps. N(0, 0.02) keeps things stable.
        """
        # Embedding: small random init
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Re-zero the padding vector (init above overwrites it)
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

        # Linear layers: Xavier uniform
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _mean_pool(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Average embeddings, ignoring padding positions.

        Args:
            embeddings: [batch, seq_len, embed_dim] — raw embedding lookup
            mask:       [batch, seq_len] — True for real trigrams, False for padding

        Returns:
            [batch, embed_dim] — one vector per text

        WHY MASKED MEAN (not just .mean(dim=1)):
            Consider two inputs:
              "dal"           → 6 real trigrams + 42 padding
              "paneer tikka"  → 11 real trigrams + 37 padding

            Without masking: "dal" gets averaged with 42 zero vectors, making
            its representation very close to zero — weak and uninformative.
            With masking: "dal" is the average of 6 meaningful vectors — strong
            and expressive. This lets short and long texts have equal "power".
        """
        # Expand mask to match embedding dimensions: [batch, seq, 1]
        mask_expanded = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

        # Zero out padding embeddings (belt-and-suspenders with padding_idx)
        masked_embeddings = embeddings * mask_expanded

        # Sum along sequence dimension
        summed = masked_embeddings.sum(dim=1)  # [batch, embed_dim]

        # Count non-padding tokens per example (avoid division by zero)
        counts = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]

        return summed / counts

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: token IDs → L2-normalized embedding.

        Args:
            token_ids: [batch_size, max_trigrams] — integer tensor from tokenizer

        Returns:
            [batch_size, embed_dim] — unit-length embedding vectors

        The pipeline:
            token_ids → embedding lookup → masked mean pool → MLP → L2 norm
        """
        # Step 1: Create padding mask (True = real token, False = padding)
        mask = token_ids != self.pad_id  # [batch, seq_len]

        # Step 2: Embedding lookup
        # Each ID pulls its row from the 25K × 80 table
        embeds = self.embedding(token_ids)  # [batch, seq_len, embed_dim]

        # Step 3: Mean pooling over non-padding positions
        pooled = self._mean_pool(embeds, mask)  # [batch, embed_dim]

        # Step 4: MLP projection
        projected = self.projection(pooled)  # [batch, embed_dim]

        # Step 5: L2 normalize — project onto unit sphere
        # After this, dot product = cosine similarity (no division needed)
        normalized = F.normalize(projected, p=2, dim=-1)  # [batch, embed_dim]

        return normalized


# ---------------------------------------------------------------------------
# Quick self-test — run with: python -m model.encoder
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from model.tokenizer import CharTrigramTokenizer

    tok = CharTrigramTokenizer()
    enc = TextEncoder(num_buckets=tok.num_buckets, embed_dim=128)

    # Count parameters
    total_params = sum(p.numel() for p in enc.parameters())
    trainable_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Model size (MB):  {total_params * 4 / 1024 / 1024:.1f} (float32)")
    print()

    # Encode some texts
    texts = [
        "paneer tikka",
        "paner tika",          # typo version
        "butter chicken",
        "gulab jamun",         # completely different dish
    ]

    # Tokenize
    ids = torch.tensor([tok.encode(t) for t in texts])
    print(f"Input shape:  {ids.shape}")  # [4, 48]

    # Encode
    enc.eval()
    with torch.no_grad():
        embeddings = enc(ids)
    print(f"Output shape: {embeddings.shape}")  # [4, 128]

    # Check L2 norm (should be 1.0 for all)
    norms = embeddings.norm(dim=-1)
    print(f"L2 norms:     {norms.tolist()}")

    # Compute pairwise cosine similarities
    # (since vectors are unit-length, dot product = cosine similarity)
    sim_matrix = embeddings @ embeddings.T
    print(f"\nCosine similarity matrix:")
    print(f"{'':20s} ", end="")
    for t in texts:
        print(f"{t:>18s}", end="")
    print()
    for i, t in enumerate(texts):
        print(f"{t:20s} ", end="")
        for j in range(len(texts)):
            print(f"{sim_matrix[i][j].item():18.3f}", end="")
        print()

    print(f"\nNote: similarities are random (untrained model).")
    print(f"After training, 'paneer tikka' and 'paner tika' should be close (~0.9+).")
