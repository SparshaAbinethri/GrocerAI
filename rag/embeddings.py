"""
rag/embeddings.py
Embedding utilities for the GrocerAI RAG preference store.
Provides a thin wrapper around OpenAI embeddings with:
  - Batching for large document sets
  - Cosine similarity helper
  - Text preprocessing for grocery/preference domain
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

from langchain_openai import OpenAIEmbeddings

from core.config import settings

logger = logging.getLogger(__name__)


# ─── Singleton embedder ───────────────────────────────────────────────────────

_embedder: OpenAIEmbeddings | None = None


def get_embedder() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance."""
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        logger.debug("Initialised OpenAIEmbeddings (model=%s)", settings.embedding_model)
    return _embedder


# ─── Text preprocessing ───────────────────────────────────────────────────────

def preprocess_preference_text(text: str) -> str:
    """
    Normalise preference text before embedding.
    Lowercases, strips extra whitespace, expands common abbreviations.
    """
    text = text.lower().strip()
    # Expand common grocery shorthand
    replacements = {
        "gf": "gluten-free",
        "df": "dairy-free",
        "v ": "vegan ",
        "veg ": "vegetarian ",
        "org ": "organic ",
    }
    for short, full in replacements.items():
        text = text.replace(short, full)
    return " ".join(text.split())  # collapse whitespace


def preprocess_grocery_item(item: str) -> str:
    """
    Normalise a grocery item name for consistent embedding.
    e.g. "2% Milk" → "milk 2 percent", "OJ" → "orange juice"
    """
    item = item.lower().strip()
    expansions = {
        "oj": "orange juice",
        "pb": "peanut butter",
        "evoo": "extra virgin olive oil",
        "2%": "two percent",
        "1%": "one percent",
    }
    for abbr, full in expansions.items():
        if item == abbr or item.startswith(abbr + " "):
            item = item.replace(abbr, full, 1)
    return item


# ─── Similarity helpers ───────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Returns a float in [-1, 1]; higher = more similar.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def embed_texts(texts: Sequence[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed a list of texts, batching to stay within API limits.
    Returns a list of embedding vectors in the same order as input.
    """
    embedder = get_embedder()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        logger.debug("Embedding batch %d–%d of %d", i, i + len(batch), len(texts))
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string (uses query-optimised embedding)."""
    return get_embedder().embed_query(preprocess_preference_text(query))


# ─── Preference relevance scoring ─────────────────────────────────────────────

def score_preference_relevance(
    item_name: str,
    preference_texts: list[str],
    threshold: float = 0.75,
) -> list[tuple[str, float]]:
    """
    Score how relevant a grocery item is to a list of preference texts.
    Returns list of (preference_text, similarity_score) above threshold,
    sorted by score descending.

    Useful for: checking if a new grocery item matches any stored dietary rule.

    Example:
        score_preference_relevance("almond milk", ["dairy-free preference", "nut allergy"])
        → [("dairy-free preference", 0.82), ("nut allergy", 0.61)]
    """
    if not preference_texts:
        return []

    item_vec = embed_query(item_name)
    pref_vecs = embed_texts([preprocess_preference_text(p) for p in preference_texts])

    scored = [
        (pref, cosine_similarity(item_vec, pref_vec))
        for pref, pref_vec in zip(preference_texts, pref_vecs)
    ]

    return sorted(
        [(pref, score) for pref, score in scored if score >= threshold],
        key=lambda x: x[1],
        reverse=True,
    )
