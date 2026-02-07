# Dual-encoder training: load pairs → InfoNCE loss → validate by R@K. Best checkpoint by val R@5.
# AdamW, LR 3e-4 with warmup + cosine decay.

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from model.tokenizer import CharTrigramTokenizer
from model.scorer import DualEncoderScorer
from training.dataset import create_dataloaders, QueryDishDataset, get_dish_display_text
from training.loss import InfoNCELoss


# ===================================================================
# Evaluation metrics
# ===================================================================

@torch.no_grad()
def compute_full_catalog_recall(
    model: nn.Module,
    val_dataset: QueryDishDataset,
    all_dish_names: list[str],
    tokenizer: CharTrigramTokenizer,
    k_values: list[int] = [1, 5, 10],
    encode_batch_size: int = 512,
    dish_texts_for_encoding: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute Recall@K by ranking each val query against the FULL dish catalog.

    THIS IS THE PRODUCTION-REALISTIC METRIC.

    dish_texts_for_encoding: if set (e.g. "name | diet | course"), encode these
    instead of all_dish_names; indexing and correct-dish lookup still use names.
    """
    model.eval()

    texts_to_encode = dish_texts_for_encoding if dish_texts_for_encoding is not None else all_dish_names
    # Step 1: Encode full dish catalog in batches
    dish_ids = torch.tensor(
        tokenizer.encode_batch(texts_to_encode), dtype=torch.long
    )
    dish_emb_chunks = []
    for start in range(0, len(dish_ids), encode_batch_size):
        chunk = dish_ids[start : start + encode_batch_size]
        dish_emb_chunks.append(model.encoder(chunk))
    dish_embs = torch.cat(dish_emb_chunks, dim=0)  # [num_dishes, embed_dim]

    # Step 2: Build correct-dish index for each val query
    dish_to_idx = {name: i for i, name in enumerate(all_dish_names)}
    correct_indices = []
    valid_mask = []
    for i in range(len(val_dataset)):
        _, dish_text = val_dataset.get_raw_pair(i)
        if dish_text in dish_to_idx:
            correct_indices.append(dish_to_idx[dish_text])
            valid_mask.append(True)
        else:
            correct_indices.append(0)  # placeholder
            valid_mask.append(False)

    correct_indices = torch.tensor(correct_indices, dtype=torch.long)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    total = valid_mask.sum().item()

    if total == 0:
        metrics = {f"recall@{k}": 0.0 for k in k_values}
        metrics["mrr"] = 0.0
        metrics["total_queries"] = 0
        metrics["catalog_size"] = len(all_dish_names)
        model.train()
        return metrics

    # Step 3: Encode all val queries in batches
    q_emb_chunks = []
    for start in range(0, len(val_dataset), encode_batch_size):
        chunk = val_dataset.query_ids[start : start + encode_batch_size]
        q_emb_chunks.append(model.encoder(chunk))
    all_q_embs = torch.cat(q_emb_chunks, dim=0)  # [N_val, embed_dim]

    # Step 4: Vectorized ranking
    # Compute score of each query's correct dish
    correct_scores = (all_q_embs * dish_embs[correct_indices]).sum(dim=1)  # [N_val]

    # Count how many dishes score higher than the correct dish per query
    # Process in chunks to limit memory: [chunk, num_dishes]
    correct_at_k = {k: 0 for k in k_values}
    rr_sum = 0.0

    for start in range(0, len(all_q_embs), encode_batch_size):
        end = min(start + encode_batch_size, len(all_q_embs))
        chunk_embs = all_q_embs[start:end]              # [chunk, embed_dim]
        chunk_scores = chunk_embs @ dish_embs.T          # [chunk, num_dishes]
        chunk_correct = correct_scores[start:end]        # [chunk]
        chunk_valid = valid_mask[start:end]               # [chunk]

        # rank = number of dishes with strictly higher score (0-indexed)
        ranks = (chunk_scores > chunk_correct.unsqueeze(1)).sum(dim=1)  # [chunk]

        for i_local in range(end - start):
            if not chunk_valid[i_local]:
                continue
            rank = ranks[i_local].item()
            for k in k_values:
                if rank < k:
                    correct_at_k[k] += 1
            rr_sum += 1.0 / (rank + 1)

    metrics = {f"recall@{k}": correct_at_k[k] / max(total, 1) for k in k_values}
    metrics["mrr"] = rr_sum / max(total, 1)
    metrics["total_queries"] = total
    metrics["catalog_size"] = len(all_dish_names)

    model.train()
    return metrics


# ===================================================================
# Per-query-type evaluation (B3)
# ===================================================================

# Hinglish markers for query type inference
_HINGLISH_WORDS = frozenset({
    "chahiye", "khana", "wala", "wali", "kuch", "aaj", "ghar",
    "meetha", "masaledar", "jaisa", "liye", "mann", "bhookh",
    "bhi", "mein", "ke", "ka", "ki", "hai", "nahi", "dedo",
    "banao", "milega", "saath", "raat", "subah", "thand",
    "garma", "garam", "thanda", "peene", "kha", "khao",
})

# Category/cuisine signal words
_CATEGORY_WORDS = frozenset({
    "indian", "cuisine", "breakfast", "dessert", "snack", "lunch",
    "dinner", "vegetarian", "food", "dish", "sweet", "spicy",
    "south", "north", "bengali", "gujarati", "maharashtrian",
    "chettinad", "andhra", "rajasthani", "continental", "chinese",
})


def infer_query_type(query: str, dish_name: str) -> str:
    """
    Heuristic classification of query-dish pairs into types.

    Why heuristic? Tracking exact types through the data pipeline
    (augmentation, dedup, split) is complex. Inference from patterns
    is good enough for evaluation diagnostics — we want to know
    roughly WHERE the model struggles, not classify every query perfectly.
    """
    q = query.lower()
    d = dish_name.lower()
    q_words = set(q.split())

    # Exact match (query is the dish name or its lowercase)
    if q == d:
        return "exact"

    # Hinglish: contains any Hindi/Hinglish words
    if q_words & _HINGLISH_WORDS:
        return "hinglish"

    # Ingredient: "something with paneer", "paneer food", "paneer dish"
    # Catches both template patterns and standalone suffix forms
    ingredient_suffixes = {"food", "dish", "wala", "wali"}
    if (any(pat in q for pat in ("something with", "dish with", "food with",
                                  "recipe with", " with "))
        or (len(q_words) >= 2 and q_words & ingredient_suffixes)):
        return "ingredient"

    # Category/cuisine: regional or categorical queries
    if len(q_words & _CATEGORY_WORDS) >= 2:
        return "category"

    # Partial name: query words overlap with dish name words
    d_words = set(d.split())
    if len(q_words & d_words) >= 1:
        return "partial"

    return "other"


@torch.no_grad()
def compute_per_type_recall(
    model: nn.Module,
    val_dataset: QueryDishDataset,
    all_dish_names: list[str],
    tokenizer: CharTrigramTokenizer,
    k_values: list[int] = [1, 5, 10],
    encode_batch_size: int = 512,
    dish_texts_for_encoding: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Break down Recall@K and MRR by inferred query type.

    Returns: { query_type: { "recall@1": ..., "count": N }, ... }
    """
    model.eval()

    texts_to_encode = dish_texts_for_encoding if dish_texts_for_encoding is not None else all_dish_names
    # Step 1: Encode full dish catalog
    dish_ids = torch.tensor(
        tokenizer.encode_batch(texts_to_encode), dtype=torch.long
    )
    dish_emb_chunks = []
    for start in range(0, len(dish_ids), encode_batch_size):
        chunk = dish_ids[start : start + encode_batch_size]
        dish_emb_chunks.append(model.encoder(chunk))
    dish_embs = torch.cat(dish_emb_chunks, dim=0)

    # Step 2: Build correct-dish index + infer types
    dish_to_idx = {name: i for i, name in enumerate(all_dish_names)}
    correct_indices = []
    valid_mask = []
    query_types = []
    for i in range(len(val_dataset)):
        q_text, d_text = val_dataset.get_raw_pair(i)
        if d_text in dish_to_idx:
            correct_indices.append(dish_to_idx[d_text])
            valid_mask.append(True)
        else:
            correct_indices.append(0)
            valid_mask.append(False)
        query_types.append(infer_query_type(q_text, d_text))

    correct_indices = torch.tensor(correct_indices, dtype=torch.long)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)

    # Step 3: Encode all val queries
    q_emb_chunks = []
    for start in range(0, len(val_dataset), encode_batch_size):
        chunk = val_dataset.query_ids[start : start + encode_batch_size]
        q_emb_chunks.append(model.encoder(chunk))
    all_q_embs = torch.cat(q_emb_chunks, dim=0)

    # Step 4: Compute correct dish scores
    correct_scores = (all_q_embs * dish_embs[correct_indices]).sum(dim=1)

    # Step 5: Compute ranks per query, grouped by type
    type_metrics = defaultdict(lambda: {"count": 0, "rr_sum": 0.0,
                                         **{f"correct@{k}": 0 for k in k_values}})

    for start in range(0, len(all_q_embs), encode_batch_size):
        end = min(start + encode_batch_size, len(all_q_embs))
        chunk_embs = all_q_embs[start:end]
        chunk_scores = chunk_embs @ dish_embs.T
        chunk_correct = correct_scores[start:end]
        chunk_valid = valid_mask[start:end]
        ranks = (chunk_scores > chunk_correct.unsqueeze(1)).sum(dim=1)

        for i_local in range(end - start):
            global_idx = start + i_local
            if not chunk_valid[i_local]:
                continue
            qtype = query_types[global_idx]
            rank = ranks[i_local].item()
            type_metrics[qtype]["count"] += 1
            type_metrics[qtype]["rr_sum"] += 1.0 / (rank + 1)
            for k in k_values:
                if rank < k:
                    type_metrics[qtype][f"correct@{k}"] += 1

    # Compute final rates
    results = {}
    for qtype, m in sorted(type_metrics.items()):
        n = m["count"]
        results[qtype] = {
            "count": n,
            "mrr": m["rr_sum"] / max(n, 1),
            **{f"recall@{k}": m[f"correct@{k}"] / max(n, 1) for k in k_values},
        }

    model.train()
    return results


def print_per_type_report(per_type: dict[str, dict[str, float]]):
    """Pretty-print per-query-type evaluation results."""
    print("\n" + "=" * 72)
    print("Per-query-type evaluation breakdown")
    print("=" * 72)
    print(f"  {'Type':<12s} {'Count':>6s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s} {'MRR':>7s}")
    print("  " + "-" * 52)
    for qtype, m in sorted(per_type.items(), key=lambda x: -x[1]["count"]):
        print(f"  {qtype:<12s} {m['count']:>6d} "
              f"{m['recall@1']:>7.3f} {m['recall@5']:>7.3f} "
              f"{m['recall@10']:>7.3f} {m['mrr']:>7.3f}")
    # Print total
    total_count = sum(m["count"] for m in per_type.values())
    print("  " + "-" * 52)
    print(f"  {'TOTAL':<12s} {total_count:>6d}")
    print("=" * 72)


# ===================================================================
# Learning rate schedule
# ===================================================================

def get_lr(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.1) -> float:
    """
    Linear warmup + cosine decay schedule.

    Steps 0 → warmup: LR ramps linearly from 0 to base_lr
    Steps warmup → end: LR follows cosine curve from base_lr to 0

    This is the standard schedule used by BERT, GPT, CLIP, etc.
    """
    warmup_steps = int(total_steps * warmup_frac)

    if step < warmup_steps:
        # Linear warmup: 0 → base_lr
        return base_lr * (step + 1) / warmup_steps
    else:
        # Cosine decay: base_lr → 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ===================================================================
# Training loop
# ===================================================================

def train(
    # Data
    train_csv: str,
    val_csv: str,
    dish_catalog_csv: str,
    # Model
    num_buckets: int = 25_000,
    embed_dim: int = 80,
    dropout: float = 0.18,
    max_trigrams: int = 48,
    # Training
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    temperature: float = 0.07,
    early_stop_patience: int = 8,
    # Output
    checkpoint_dir: str = "checkpoints",
):
    """
    Full training pipeline.

    Config: 25K buckets, 80d, dropout 0.18 (~2M params).
    Dishes encoded as "name | diet | course" when catalog available (stronger signal).
    """
    device = torch.device("cpu")
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # --- Initialize components ---
    print("=" * 60)
    print("Initializing...")
    print("=" * 60)

    tokenizer = CharTrigramTokenizer(
        num_buckets=num_buckets, max_trigrams=max_trigrams
    )

    model = DualEncoderScorer(
        num_buckets=num_buckets,
        embed_dim=embed_dim,
        dropout=dropout,
        max_trigrams=max_trigrams,
    ).to(device)

    loss_fn = InfoNCELoss(temperature=temperature).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params:  {total_params:,}")
    print(f"  Dropout:       {dropout}")
    print(f"  Device:        {device}")

    # --- Data ---
    train_loader, val_loader = create_dataloaders(
        train_csv, val_csv, tokenizer,
        batch_size=batch_size,
        dish_catalog_csv=dish_catalog_csv,
    )
    val_dataset = val_loader.dataset  # Need raw access for full-catalog eval

    # Load full dish catalog for evaluation (names + display text for encoding)
    dish_catalog = []
    with open(dish_catalog_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dish_catalog.append(row)
    all_dish_names = [r["name"] for r in dish_catalog]
    all_dish_display_texts = [get_dish_display_text(r) for r in dish_catalog]

    print(f"  Train pairs:   {len(train_loader.dataset):,}")
    print(f"  Val pairs:     {len(val_dataset):,}")
    print(f"  Dish catalog:  {len(all_dish_names)} dishes")
    print(f"  Batch size:    {batch_size}")
    print(f"  Train batches: {len(train_loader)}")

    # --- Optimizer ---
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=learning_rate, weight_decay=weight_decay
    )

    total_steps = epochs * len(train_loader)
    print(f"  Max epochs:    {epochs} (early stop patience={early_stop_patience})")
    print(f"  Total steps:   {total_steps} (max)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay:  {weight_decay}")
    print()

    # --- Training loop ---
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_recall_at_1": [],
        "val_recall_at_5": [],
        "val_recall_at_10": [],
        "val_mrr": [],
        "temperature": [],
        "lr": [],
    }
    best_recall_at_5 = 0.0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs}",
            leave=False,
            ncols=100,
        )

        for batch_idx, (query_ids, dish_ids) in enumerate(pbar):
            query_ids = query_ids.to(device)
            dish_ids = dish_ids.to(device)

            # --- Learning rate schedule ---
            lr = get_lr(global_step, total_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # --- Forward ---
            similarity = model(query_ids, dish_ids)
            loss = loss_fn(similarity)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                lr=f"{lr:.1e}",
                tau=f"{loss_fn.temperature:.3f}",
            )

        # --- Epoch stats ---
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # --- Validation: loss (batch-level, fast) ---
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for query_ids, dish_ids in val_loader:
                query_ids = query_ids.to(device)
                dish_ids = dish_ids.to(device)
                similarity = model(query_ids, dish_ids)
                val_loss_total += loss_fn(similarity).item()
                val_batches += 1
        avg_val_loss = val_loss_total / max(val_batches, 1)

        # --- Validation: full-catalog recall (production-realistic) ---
        recall = compute_full_catalog_recall(
            model, val_dataset, all_dish_names, tokenizer,
            k_values=[1, 5, 10],
            dish_texts_for_encoding=all_dish_display_texts,
        )

        # --- Log ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_recall_at_1"].append(recall["recall@1"])
        history["val_recall_at_5"].append(recall["recall@5"])
        history["val_recall_at_10"].append(recall["recall@10"])
        history["val_mrr"].append(recall["mrr"])
        history["temperature"].append(loss_fn.temperature)
        history["lr"].append(lr)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"R@1: {recall['recall@1']:.3f} | "
            f"R@5: {recall['recall@5']:.3f} | "
            f"R@10: {recall['recall@10']:.3f} | "
            f"MRR: {recall['mrr']:.3f} | "
            f"τ: {loss_fn.temperature:.3f} | "
            f"{epoch_time:.1f}s"
        )

        # --- Checkpoint + early stopping ---
        if recall["recall@5"] > best_recall_at_5:
            best_recall_at_5 = recall["recall@5"]
            epochs_without_improvement = 0
            save_checkpoint(
                model, tokenizer, loss_fn, history, epoch,
                checkpoint_path / "best_model.pt",
            )
            print(f"  → New best! R@5={best_recall_at_5:.3f} MRR={recall['mrr']:.3f} (saved)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"\n  Early stopping: no improvement for {early_stop_patience} epochs.")
                break

    # --- Save final model ---
    save_checkpoint(
        model, tokenizer, loss_fn, history, epoch,
        checkpoint_path / "final_model.pt",
    )
    print(f"\nTraining complete.")
    print(f"  Best Recall@5: {best_recall_at_5:.3f}")
    print(f"  Stopped at epoch: {epoch}")
    print(f"  Checkpoints:   {checkpoint_path}/")

    # --- B3: Per-query-type evaluation (run once at end) ---
    print("\nRunning per-query-type evaluation on best model...")
    best_ckpt = torch.load(
        checkpoint_path / "best_model.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    per_type = compute_per_type_recall(
        model, val_dataset, all_dish_names, tokenizer,
        dish_texts_for_encoding=all_dish_display_texts,
    )
    print_per_type_report(per_type)

    return model, history


# ===================================================================
# Checkpoint save / load
# ===================================================================

def save_checkpoint(model, tokenizer, loss_fn, history, epoch, path):
    """
    Save model + tokenizer config + training state.

    We save tokenizer CONFIG (not weights — it has none) so we can
    reconstruct the exact same tokenizer at inference time. The model
    and tokenizer must agree on num_buckets and max_trigrams.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "tokenizer_config": tokenizer.get_config(),
        "embed_dim": model.embed_dim,
        "temperature": loss_fn.temperature,
        "history": history,
    }, path)


def load_checkpoint(path, device="cpu"):
    """
    Load a saved checkpoint and reconstruct model + tokenizer.

    Returns:
        (model, tokenizer) ready for inference.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct tokenizer from saved config
    tokenizer = CharTrigramTokenizer.from_config(checkpoint["tokenizer_config"])

    # Reconstruct model with same architecture
    config = checkpoint["tokenizer_config"]
    model = DualEncoderScorer(
        num_buckets=config["num_buckets"],
        embed_dim=checkpoint.get("embed_dim", 64),
        max_trigrams=config["max_trigrams"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Temperature: {checkpoint['temperature']:.4f}")

    return model, tokenizer


# ===================================================================
# Main entry point
# ===================================================================

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

    model, history = train(
        train_csv=str(DATA_DIR / "train.csv"),
        val_csv=str(DATA_DIR / "val.csv"),
        dish_catalog_csv=str(DATA_DIR / "dishes.csv"),
        # Small catalog (255 dishes): smaller model to avoid overfitting
        num_buckets=8_000,
        embed_dim=48,
        dropout=0.25,
        max_trigrams=48,
        # Training config
        epochs=25,
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=0.01,
        temperature=0.07,
        early_stop_patience=5,
        # Output
        checkpoint_dir=str(Path(__file__).resolve().parent.parent / "checkpoints"),
    )
