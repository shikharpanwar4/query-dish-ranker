"""
Custom BM25 for dish catalog: no external deps (e.g. rank_bm25).

Scores query against each "document" (dish name + optional metadata).
Uses same tokenization/normalization as our lexical matcher for consistency.
"""

import math
import re
from collections import Counter


def _collapse_repeats(text: str) -> str:
    """Collapse repeated consecutive chars for typo tolerance."""
    if not text:
        return text
    out = [text[0]]
    for c in text[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop short tokens; normalize repeats for fuzzy."""
    text = (text or "").lower().strip()
    tokens = re.split(r"[^\w]+", text)
    # Use normalized form for matching (daal -> dal)
    return [_collapse_repeats(t) for t in tokens if len(t) >= 2]


class BM25:
    """
    BM25 over a fixed set of documents (dish texts).
    IDF computed from document collection; k1 and b are standard BM25 params.
    """

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_tokens = [tokenize(d) for d in documents]
        self.n_docs = len(documents)
        self.avgdl = (
            sum(len(toks) for toks in self.doc_tokens) / self.n_docs
            if self.n_docs else 0
        )
        # Document frequency: df[t] = number of docs containing t
        self.df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        # IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {}
        for t, df in self.df.items():
            self.idf[t] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """
        Return BM25 score for each document for the given query.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return [0.0] * self.n_docs

        scores = []
        for doc_tokens in self.doc_tokens:
            doc_len = len(doc_tokens)
            if doc_len == 0:
                scores.append(0.0)
                continue
            tf = Counter(doc_tokens)
            s = 0.0
            for t in q_tokens:
                if t not in self.idf:
                    continue
                f = tf.get(t, 0)
                # BM25 term weight: idf * (f * (k1+1)) / (f + k1 * (1 - b + b * doc_len/avgdl))
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                s += self.idf[t] * (f * (self.k1 + 1)) / denom
            scores.append(s)
        return scores
