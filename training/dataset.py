# (query, dish_name) pairs; negatives from in-batch (InfoNCE). Query typos at train time; dish = rich text (name | diet | course | ...).

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from model.tokenizer import CharTrigramTokenizer


def get_dish_display_text(row: dict) -> str:
    """name | diet | course | flavor | region | state | style | category | first 5 ingredients."""
    name = (row.get("name") or "").strip()
    diet = (row.get("diet") or "").strip()
    course = (row.get("course") or "").strip()
    flavor = (row.get("flavor") or "").strip()
    region = (row.get("region") or "").strip()
    state = (row.get("state") or "").strip()
    style = (row.get("style") or "").strip()
    category = (row.get("category") or "").strip()
    parts = [name]
    if diet:
        parts.append(diet)
    if course:
        parts.append(course)
    if flavor:
        parts.append(flavor)
    if region:
        parts.append(region)
    if state:
        parts.append(state)
    if style:
        parts.append(style)
    if category:
        parts.append(category)
    # Short ingredient snippet (first 5) so "paneer" / "chicken" in query can match
    ingredients = row.get("ingredients")
    if ingredients:
        if isinstance(ingredients, str):
            ing_list = [x.strip() for x in ingredients.split(",") if x.strip()][:5]
        else:
            ing_list = list(ingredients)[:5]
        if ing_list:
            parts.append(" ".join(ing_list))
    s = " | ".join(parts).strip()
    return s if s else name


def _inject_typo_online(text: str) -> str:
    """One random typo (swap/delete/duplicate) in a word 3+ chars."""
    words = text.split()
    if not words:
        return text

    # Pick a random word with 3+ chars
    long_words = [(i, w) for i, w in enumerate(words) if len(w) >= 3]
    if not long_words:
        return text

    idx, word = random.choice(long_words)
    chars = list(word)

    if len(chars) < 3:
        return text

    op = random.choice(["swap", "delete", "duplicate"])
    pos = random.randint(1, len(chars) - 2)

    if op == "swap" and pos < len(chars) - 1:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    elif op == "delete":
        chars.pop(pos)
    elif op == "duplicate":
        chars.insert(pos, chars[pos])

    words[idx] = "".join(chars)
    return " ".join(words)


class QueryDishDataset(Dataset):
    """(query, dish) pairs; in-batch negatives for InfoNCE. Query typos at getitem; dish = rich text."""

    def __init__(
        self,
        csv_path: str,
        tokenizer: CharTrigramTokenizer,
        augment: bool = False,
        augment_prob: float = 0.5,
        dish_catalog_csv: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.augment = augment
        self.augment_prob = augment_prob
        self.queries: list[str] = []
        self.dishes: list[str] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row["query"].strip()
                d = row["dish_name"].strip()
                if q and d:
                    self.queries.append(q)
                    self.dishes.append(d)

        self.name_to_display: dict[str, str] = {}
        if dish_catalog_csv and Path(dish_catalog_csv).exists():
            with open(dish_catalog_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = (row.get("name") or "").strip()
                    self.name_to_display[name] = get_dish_display_text(row)

        dish_texts = [self.name_to_display.get(d, d) for d in self.dishes]
        self.dish_ids = torch.tensor(tokenizer.encode_batch(dish_texts), dtype=torch.long)
        self.query_ids = torch.tensor(tokenizer.encode_batch(self.queries), dtype=torch.long)
        assert len(self.query_ids) == len(self.dish_ids)

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        dish_ids = self.dish_ids[idx]
        if self.augment and random.random() < self.augment_prob:
            query_text = _inject_typo_online(self.queries[idx])
            if random.random() < 0.1:
                query_text = query_text.upper()
            elif random.random() < 0.05:
                query_text = query_text.title()
            query_ids = torch.tensor(self.tokenizer.encode(query_text), dtype=torch.long)
        else:
            query_ids = self.query_ids[idx]
        return query_ids, dish_ids

    def get_raw_pair(self, idx: int) -> tuple[str, str]:
        """Get original text pair for debugging / display."""
        return self.queries[idx], self.dishes[idx]


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    tokenizer: CharTrigramTokenizer,
    batch_size: int = 128,
    num_workers: int = 0,
    dish_catalog_csv: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Train has augment=True (typos); val has augment=False for stable metrics."""
    train_dataset = QueryDishDataset(
        train_csv, tokenizer, augment=True, augment_prob=0.3, dish_catalog_csv=dish_catalog_csv
    )
    val_dataset = QueryDishDataset(
        val_csv, tokenizer, augment=False, dish_catalog_csv=dish_catalog_csv
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick self-test — run with: python -m training.dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    TRAIN_CSV = DATA_DIR / "train.csv"
    VAL_CSV = DATA_DIR / "val.csv"

    tok = CharTrigramTokenizer()

    print("Loading datasets...")
    train_ds = QueryDishDataset(TRAIN_CSV, tok)
    val_ds = QueryDishDataset(VAL_CSV, tok)

    print(f"  Train: {len(train_ds):,} pairs")
    print(f"  Val:   {len(val_ds):,} pairs")
    print(f"  Query tensor shape: {train_ds.query_ids.shape}")
    print(f"  Dish tensor shape:  {train_ds.dish_ids.shape}")
    mem_mb = (train_ds.query_ids.nbytes + train_ds.dish_ids.nbytes) / 1024 / 1024
    print(f"  Memory: {mem_mb:.1f} MB")

    # Test a few samples
    print("\n  Sample pairs:")
    for i in [0, 100, 500]:
        q, d = train_ds.get_raw_pair(i)
        q_ids, d_ids = train_ds[i]
        non_pad_q = (q_ids != 0).sum().item()
        non_pad_d = (d_ids != 0).sum().item()
        print(f"    [{i}] Q: '{q}' ({non_pad_q} trigrams) → D: '{d}' ({non_pad_d} trigrams)")

    # Test DataLoader
    print("\nCreating DataLoaders (batch_size=128)...")
    train_loader, val_loader = create_dataloaders(
        TRAIN_CSV, VAL_CSV, tok, batch_size=128
    )
    print(f"  Train batches: {len(train_loader)} (drop_last=True)")
    print(f"  Val batches:   {len(val_loader)} (drop_last=False)")

    # Inspect one batch
    batch_q, batch_d = next(iter(train_loader))
    print(f"\n  Batch shapes:")
    print(f"    query_ids: {batch_q.shape}")
    print(f"    dish_ids:  {batch_d.shape}")
    print(f"    dtype:     {batch_q.dtype}")
