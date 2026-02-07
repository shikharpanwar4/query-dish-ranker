# Char trigrams + feature hashing. Typo-tolerant (similar strings share trigrams); no vocab file.
import hashlib
from typing import Union


class CharTrigramTokenizer:
    """Stateless: text → trigram IDs via hashing. No vocab; pad_id=0."""

    def __init__(
        self,
        num_buckets: int = 25_000,
        ngram_size: int = 3,
        max_trigrams: int = 48,
        pad_id: int = 0,
    ):
        self.num_buckets = num_buckets
        self.ngram_size = ngram_size
        self.max_trigrams = max_trigrams
        self.pad_id = pad_id

    def _extract_trigrams(self, text: str) -> list[str]:
        """Words → #word# → sliding trigrams."""
        text = text.lower().strip()
        trigrams = []
        for word in text.split():
            padded = f"#{word}#"

            for i in range(len(padded) - self.ngram_size + 1):
                trigrams.append(padded[i : i + self.ngram_size])

        return trigrams

    def _hash_trigram(self, trigram: str) -> int:
        """MD5-based bucket in [1, num_buckets-1]; 0 reserved for pad."""
        h = hashlib.md5(trigram.encode("utf-8")).hexdigest()
        return (int(h[:8], 16) % (self.num_buckets - 1)) + 1

    def encode(self, text: str) -> list[int]:
        """Text → trigrams → bucket IDs → pad/truncate to max_trigrams."""
        trigrams = self._extract_trigrams(text)
        ids = [self._hash_trigram(t) for t in trigrams]

        # Truncate if too long, pad if too short
        if len(ids) > self.max_trigrams:
            ids = ids[: self.max_trigrams]
        else:
            ids = ids + [self.pad_id] * (self.max_trigrams - len(ids))

        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a list of texts. Useful for batch processing."""
        return [self.encode(t) for t in texts]

    def get_config(self) -> dict:
        """Return config dict for serialization / reconstruction."""
        return {
            "num_buckets": self.num_buckets,
            "ngram_size": self.ngram_size,
            "max_trigrams": self.max_trigrams,
            "pad_id": self.pad_id,
        }

    @classmethod
    def from_config(cls, config: dict) -> "CharTrigramTokenizer":
        """Reconstruct tokenizer from config dict."""
        return cls(**config)

    def debug(self, text: str) -> dict:
        """Return trigrams and IDs for debugging."""
        trigrams = self._extract_trigrams(text)
        ids = [self._hash_trigram(t) for t in trigrams]
        return {
            "input": text,
            "normalized": text.lower().strip(),
            "trigrams": trigrams,
            "num_trigrams": len(trigrams),
            "hash_ids": ids,
            "padded_length": self.max_trigrams,
        }


# ---------------------------------------------------------------------------
# Quick self-test — run with: python -m model.tokenizer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tok = CharTrigramTokenizer()

    # --- Demo: basic tokenization ---
    print("=== Basic tokenization ===")
    for text in ["paneer tikka", "Paneer Tikka", "paner tika", "PANEER TIKKA"]:
        info = tok.debug(text)
        print(f"\n  Input:    '{text}'")
        print(f"  Trigrams: {info['trigrams']}")
        print(f"  Count:    {info['num_trigrams']}")
        print(f"  IDs:      {info['hash_ids'][:8]}...")

    # --- Demo: typo tolerance ---
    print("\n=== Typo tolerance (trigram overlap) ===")
    pairs = [
        ("paneer tikka", "paner tika"),
        ("chicken biryani", "chiken biriyani"),
        ("gulab jamun", "gulba jamun"),
        ("butter chicken", "butr chiken"),
    ]
    for original, typo in pairs:
        tri_orig = set(tok._extract_trigrams(original))
        tri_typo = set(tok._extract_trigrams(typo))
        overlap = tri_orig & tri_typo
        union = tri_orig | tri_typo
        jaccard = len(overlap) / len(union) if union else 0
        print(f"\n  '{original}' vs '{typo}'")
        print(f"  Overlap: {len(overlap)}/{len(union)} trigrams (Jaccard={jaccard:.2f})")

    # --- Demo: encoding output ---
    print("\n=== Encoded output ===")
    encoded = tok.encode("paneer tikka masala")
    print(f"  Length: {len(encoded)} (max_trigrams={tok.max_trigrams})")
    print(f"  Non-pad: {sum(1 for x in encoded if x != 0)}")
    print(f"  IDs: {encoded[:15]}...")
