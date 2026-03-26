"""
rag/preference_store.py
FAISS-backed preference store for persisting and retrieving user preferences.

Fixes applied:
  - OpenAIEmbeddings initialized LAZILY (only when FAISS is actually needed)
  - get_preferences() uses JSON only — never touches embeddings
  - Stale embeddings deleted + re-embedded on save
  - Thread-safe atomic file write
  - Explicit conflict resolution (avoid_brands wins, vegan implies dairy-free)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from core.config import settings

logger = logging.getLogger(__name__)

_save_lock = threading.Lock()


def _pref_to_doc(user_id: str, prefs: dict[str, Any]) -> str:
    parts = [f"User preferences for {user_id}:"]
    if prefs.get("preferred_brands"):
        parts.append(f"Preferred brands: {', '.join(prefs['preferred_brands'])}")
    if prefs.get("avoid_brands"):
        parts.append(f"Avoid brands: {', '.join(prefs['avoid_brands'])}")
    if prefs.get("dietary_restrictions"):
        parts.append(f"Dietary restrictions: {', '.join(prefs['dietary_restrictions'])}")
    if prefs.get("quality_notes"):
        parts.append(f"Quality preferences: {prefs['quality_notes']}")
    if prefs.get("notes"):
        parts.append(f"Additional notes: {prefs['notes']}")
    return "\n".join(parts)


class PreferenceStore:
    """
    Manages user preferences.
    JSON file = source of truth for fast lookups (no API key needed).
    FAISS = semantic search (lazy, only initialized when needed).
    """

    def __init__(self) -> None:
        self._index_path = settings.faiss_index_path
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._prefs_file = self._index_path / "preferences.json"
        # Lazy — not initialized here so startup never fails due to missing key
        self._embeddings = None
        self._store = None
        self._raw_prefs: dict[str, dict] = self._load_raw_prefs()

    def _get_embeddings(self):
        """Lazy-initialize OpenAIEmbeddings only when FAISS is needed."""
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=api_key,
            )
        return self._embeddings

    # ── Raw JSON preferences ──────────────────────────────────────────────────

    def _load_raw_prefs(self) -> dict[str, dict]:
        if self._prefs_file.exists():
            try:
                return json.loads(self._prefs_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load preferences file: %s", exc)
        return {}

    def _save_raw_prefs(self) -> None:
        with _save_lock:
            tmp = self._prefs_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._raw_prefs, indent=2))
            tmp.replace(self._prefs_file)

    # ── FAISS index ───────────────────────────────────────────────────────────

    def _load_or_create_store(self):
        from langchain_community.vectorstores import FAISS
        if self._store is not None:
            return self._store

        index_file = self._index_path / "index.faiss"
        embeddings = self._get_embeddings()

        if index_file.exists():
            try:
                self._store = FAISS.load_local(
                    str(self._index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded FAISS index from %s", self._index_path)
                return self._store
            except Exception as exc:
                logger.warning("Failed to load FAISS index: %s — creating new", exc)

        self._store = FAISS.from_texts(
            ["GrocerAI preference store initialised"],
            embeddings,
            metadatas=[{"type": "system", "user_id": "system"}],
        )
        self._store.save_local(str(self._index_path))
        logger.info("Created new FAISS index at %s", self._index_path)
        return self._store

    def _rebuild_user_embedding(self, user_id: str, prefs: dict) -> None:
        try:
            store = self._load_or_create_store()
            docstore = store.docstore._dict
            ids_to_delete = [
                doc_id for doc_id, doc in docstore.items()
                if doc.metadata.get("user_id") == user_id
            ]
            if ids_to_delete:
                store.delete(ids_to_delete)
            doc_text = _pref_to_doc(user_id, prefs)
            store.add_texts(
                [doc_text],
                metadatas=[{"user_id": user_id, "type": "preferences"}],
            )
            store.save_local(str(self._index_path))
        except Exception as exc:
            logger.warning("FAISS update failed (prefs still saved to JSON): %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_preferences(self, user_id: str) -> dict[str, Any]:
        """Fast JSON lookup — never touches embeddings or FAISS."""
        prefs = self._raw_prefs.get(user_id, _default_preferences())
        logger.debug(
            "Preferences retrieved | user=%s brands=%s dietary=%s",
            user_id,
            prefs.get("preferred_brands", []),
            prefs.get("dietary_restrictions", []),
        )
        return prefs

    def save_preferences(self, user_id: str, prefs: dict[str, Any]) -> None:
        merged = {**_default_preferences(), **prefs}
        merged = _resolve_preference_conflicts(merged)
        self._raw_prefs[user_id] = merged
        self._save_raw_prefs()
        self._rebuild_user_embedding(user_id, merged)
        logger.info("Saved preferences for user %s", user_id)

    def search_similar_preferences(self, query: str, k: int | None = None) -> list[dict]:
        store = self._load_or_create_store()
        k = k or settings.rag_top_k
        docs = store.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def update_preference(self, user_id: str, key: str, value: Any) -> None:
        prefs = self.get_preferences(user_id)
        prefs[key] = value
        self.save_preferences(user_id, prefs)

    def add_brand_preference(self, user_id: str, brand: str, prefer: bool = True) -> None:
        prefs = self.get_preferences(user_id)
        brand_lower = brand.lower()
        if prefer:
            brands = set(prefs.get("preferred_brands", []))
            brands.add(brand_lower)
            prefs["preferred_brands"] = sorted(brands)
            prefs["avoid_brands"] = [b for b in prefs.get("avoid_brands", []) if b != brand_lower]
        else:
            brands = set(prefs.get("avoid_brands", []))
            brands.add(brand_lower)
            prefs["avoid_brands"] = sorted(brands)
            prefs["preferred_brands"] = [b for b in prefs.get("preferred_brands", []) if b != brand_lower]
        self.save_preferences(user_id, prefs)

    def add_dietary_restriction(self, user_id: str, restriction: str) -> None:
        prefs = self.get_preferences(user_id)
        restrictions = set(prefs.get("dietary_restrictions", []))
        restrictions.add(restriction.lower())
        prefs["dietary_restrictions"] = sorted(restrictions)
        self.save_preferences(user_id, prefs)


def _default_preferences() -> dict[str, Any]:
    return {
        "preferred_brands": [],
        "avoid_brands": [],
        "dietary_restrictions": [],
        "quality_notes": "",
        "notes": "",
    }


def _resolve_preference_conflicts(prefs: dict[str, Any]) -> dict[str, Any]:
    dietary = set(prefs.get("dietary_restrictions", []))
    preferred = set(prefs.get("preferred_brands", []))
    avoid = set(prefs.get("avoid_brands", []))

    # avoid always wins over preferred
    conflicts = preferred & avoid
    if conflicts:
        logger.info("Removing from preferred (also in avoid): %s", conflicts)
        preferred -= conflicts

    # vegan implies dairy-free and vegetarian
    if "vegan" in dietary:
        dietary.add("dairy-free")
        dietary.add("vegetarian")

    prefs["preferred_brands"] = sorted(preferred)
    prefs["avoid_brands"] = sorted(avoid)
    prefs["dietary_restrictions"] = sorted(dietary)
    return prefs