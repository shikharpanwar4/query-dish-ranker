"""
InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

THE CORE IDEA:
  Given a batch of N (query, dish) positive pairs, treat the task as
  N-way classification: for each query, identify its correct dish among
  all N dishes in the batch. The other N-1 dishes are "in-batch negatives".

  This is the same loss used by:
    - CLIP (OpenAI) — image-text matching
    - SimCLR (Google) — visual representation learning
    - DPR (Facebook) — dense passage retrieval

WHY THIS LOSS:
  1. No explicit negatives needed — the batch provides them for free
  2. With batch size 256, each query sees 255 negatives per step
  3. Directly optimizes ranking: "correct dish should score highest"
  4. Simple to implement: one matrix multiply + cross entropy

THE TEMPERATURE PARAMETER (τ):
  Cosine similarity ranges from -1 to 1. Softmax over these small values
  gives near-uniform probabilities → weak gradients → slow learning.

  Dividing by τ (e.g., 0.07) scales the range to roughly [-14, +14].
  Now softmax is peaky: the highest-scoring pair dominates. This makes
  training sharper and faster.

  - τ too high (0.5): soft probabilities, model doesn't commit → slow
  - τ too low (0.01): extremely sharp, model overfits to hard examples
  - τ = 0.05-0.07: good balance for our data size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss.

    "Symmetric" means we compute the loss in both directions:
      1. For each query, find the correct dish (row-wise)
      2. For each dish, find the correct query (column-wise)
    Then average the two. This provides more gradient signal per batch
    and ensures the embedding space works well in both directions.

    Without symmetry, dishes that happen to be in many batches might
    only get "pulled toward" queries but never "push away" from wrong
    queries. Symmetric loss balances both forces.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Scaling factor for similarities before softmax.
                         Lower = sharper, higher = softer.
                         0.07 is standard for contrastive learning.
        """
        super().__init__()
        # log_temperature is a learnable parameter — the model can adjust
        # temperature during training if 0.07 isn't optimal.
        # We store log(temperature) and exponentiate to ensure positivity.
        # Starting value: log(0.07) ≈ -2.66
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature).log()
        )

    @property
    def temperature(self) -> float:
        """Current temperature value (always positive via exp)."""
        return self.log_temperature.exp().item()

    def forward(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss from a similarity matrix.

        Args:
            similarity_matrix: [N, N] tensor where entry [i,j] is the
                cosine similarity between query_i and dish_j.
                Diagonal entries [i,i] are positive pairs.

        Returns:
            Scalar loss value.

        Step by step:
            1. Scale similarities by learned temperature
            2. Row-wise cross entropy: each query classifies its dish
            3. Column-wise cross entropy: each dish classifies its query
            4. Average the two
        """
        batch_size = similarity_matrix.shape[0]

        # Scale by temperature (clamped to prevent extreme values)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        logits = similarity_matrix / temperature

        # Labels: diagonal is the correct class for each row/column
        # query_0's correct dish is dish_0, query_1's is dish_1, etc.
        labels = torch.arange(batch_size, device=logits.device)

        # Row-wise loss: for each query, which dish is correct?
        loss_q2d = F.cross_entropy(logits, labels)

        # Column-wise loss: for each dish, which query is correct?
        # (transpose the matrix so columns become rows)
        loss_d2q = F.cross_entropy(logits.T, labels)

        # Symmetric loss: average of both directions
        loss = (loss_q2d + loss_d2q) / 2.0

        return loss


# ---------------------------------------------------------------------------
# Quick self-test — run with: python -m training.loss
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loss_fn = InfoNCELoss(temperature=0.07)

    print(f"Initial temperature: {loss_fn.temperature:.4f}")
    print(f"Learnable params: {sum(p.numel() for p in loss_fn.parameters())}")
    print()

    # --- Test 1: Perfect similarity matrix (diagonal = 1, off-diag = 0) ---
    print("=== Perfect predictions (loss should be very low) ===")
    perfect_sim = torch.eye(4)  # identity matrix = perfect matching
    loss = loss_fn(perfect_sim)
    print(f"  Similarity matrix:\n{perfect_sim}")
    print(f"  Loss: {loss.item():.4f}")

    # --- Test 2: Random similarity matrix (loss should be ~log(N)) ---
    print("\n=== Random predictions (loss should be ~log(4) ≈ 1.386) ===")
    random_sim = torch.randn(4, 4) * 0.1
    loss = loss_fn(random_sim)
    print(f"  Similarity matrix:\n{random_sim.round(decimals=3)}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Expected (random): ~{torch.tensor(4.0).log().item():.4f}")

    # --- Test 3: Worst case (anti-diagonal is high) ---
    print("\n=== Adversarial (correct pairs have lowest scores) ===")
    bad_sim = torch.ones(4, 4) * 0.8
    bad_sim.fill_diagonal_(0.1)  # diagonal is the WORST scoring
    loss = loss_fn(bad_sim)
    print(f"  Similarity matrix:\n{bad_sim}")
    print(f"  Loss: {loss.item():.4f}  (should be high)")

    # --- Test 4: Gradient flows ---
    print("\n=== Gradient check ===")
    sim = torch.randn(8, 8, requires_grad=True)
    loss = loss_fn(sim)
    loss.backward()
    print(f"  Gradient shape: {sim.grad.shape}")
    print(f"  Gradient norm:  {sim.grad.norm().item():.4f}")
    print(f"  Temperature grad: {loss_fn.log_temperature.grad.item():.4f}")
    print(f"  ✓ Gradients flow correctly")
