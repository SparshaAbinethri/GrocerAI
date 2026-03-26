"""
rag/preference_store.py
FAISS-backed preference store for persisting and retrieving user preferences.

Fixes applied:
  - Stale embeddings: delete + re-embed on preference update (no contradictory chunks)
  - Metadata filtering on retrieval to ensure correct user's prefs are returned
  - Retrieval logging for verification
  - Structured preference schema with explicit override logic for conflicts
  - Thread-safe save with file lock to prevent corruption on concurrent writes
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from core.config import settings

logger = logging.getLogger(__name__)

_save_lock = threading.Lock()


# ─── Preference document templates ───────────────────────────────────────────

def _pref_to_doc(user_id: str, prefs: dict[str, Any]) -> str:
    """Serialise user preferences into a searchable text document."""
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


# ─── Preference Store ─────────────────────────────────────────────────────────

class PreferenceStore:
    """
    Manages user preferences using FAISS for semantic retrieval.

    Storage layout:
      {faiss_index_path}/
        index.faiss       — FAISS binary index
        index.pkl         — docstore + metadata
        preferences.json  — raw JSON preferences keyed by user_id (source of truth)

    The JSON file is the source of truth for fast exact lookups.
    FAISS is used only for semantic similarity search across all users.
    """

    def __init__(self) -> None:
        self._index_path = settings.faiss_index_path
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._prefs_file = self._index_path / "preferences.json"
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._store: FAISS | None = None
        self._raw_prefs: dict[str, dict] = self._load_raw_prefs()

    # ── Raw JSON preferences (fast lookup — source of truth) ─────────────────

    def _load_raw_prefs(self) -> dict[str, dict]:
        if self._prefs_file.exists():
            try:
                return json.loads(self._prefs_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load preferences file: %s", exc)
        return {}

    def _save_raw_prefs(self) -> None:
        """Thread-safe write to preferences JSON file."""
        with _save_lock:
            # Write to temp file first, then rename (atomic on most filesystems)
            tmp = self._prefs_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._raw_prefs, indent=2))
            tmp.replace(self._prefs_file)

    # ── FAISS index (semantic search across users) ────────────────────────────

    def _load_or_create_store(self) -> FAISS:
        if self._store is not None:
            return self._store

        index_file = self._index_path / "index.faiss"
        if index_file.exists():
            try:
                self._store = FAISS.load_local(
                    str(self._index_path),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded FAISS index from %s", self._index_path)
                return self._store
            except Exception as exc:
                logger.warning("Failed to load FAISS index: %s — creating new", exc)

        # Create a minimal store with a placeholder document
        self._store = FAISS.from_texts(
            ["GrocerAI preference store initialised"],
            self._embeddings,
            metadatas=[{"type": "system", "user_id": "system"}],
        )
        self._store.save_local(str(self._index_path))
        logger.info("Created new FAISS index at %s", self._index_path)
        return self._store

    def _rebuild_user_embedding(self, user_id: str, prefs: dict) -> None:
        """
        Delete all existing embeddings for a user and re-embed fresh.
        This prevents stale/contradictory preference chunks from persisting.
        """
        store = self._load_or_create_store()

        # Find and delete all existing docs for this user
        try:
            docstore = store.docstore._dict
            ids_to_delete = [
                doc_id for doc_id, doc in docstore.items()
                if doc.metadata.get("user_id") == user_id
            ]
            if ids_to_delete:
                store.delete(ids_to_delete)
                logger.debug(
                    "Deleted %d stale embedding(s) for user %s",
                    len(ids_to_delete), user_id,
                )
        except Exception as exc:
            logger.warning("Could not delete stale embeddings: %s", exc)

        # Add fresh embedding
        doc_text = _pref_to_doc(user_id, prefs)
        store.add_texts(
            [doc_text],
            metadatas=[{"user_id": user_id, "type": "preferences"}],
        )
        store.save_local(str(self._index_path))

    # ── Public API ────────────────────────────────────────────────────────────

    def get_preferences(self, user_id: str) -> dict[str, Any]:
        """
        Returns user preferences dict from JSON (fast exact lookup).
        Logs what was retrieved so we can verify correctness.
        Falls back to defaults if user not found.
        """
        prefs = self._raw_prefs.get(user_id, _default_preferences())

        logger.debug(
            "Preferences retrieved | user=%s brands=%s dietary=%s",
            user_id,
            prefs.get("preferred_brands", []),
            prefs.get("dietary_restrictions", []),
        )

        return prefs

    def save_preferences(self, user_id: str, prefs: dict[str, Any]) -> None:
        """
        Persist user preferences to both JSON (fast lookup) and FAISS (semantic).
        Resolves conflicts before saving:
          - If 'vegan' in dietary, adds 'dairy-free' implicitly
          - Removes brands from preferred if they appear in avoid (avoid wins)
        """
        # Resolve conflicts before saving
        merged = {**_default_preferences(), **prefs}
        merged = _resolve_preference_conflicts(merged)

        self._raw_prefs[user_id] = merged
        self._save_raw_prefs()

        # Rebuild FAISS embedding (delete stale, add fresh)
        try:
            self._rebuild_user_embedding(user_id, merged)
        except Exception as exc:
            logger.warning("FAISS update failed (preferences still saved to JSON): %s", exc)

        logger.info(
            "Saved preferences for user %s | brands=%s dietary=%s",
            user_id,
            merged.get("preferred_brands", []),
            merged.get("dietary_restrictions", []),
        )

    def search_similar_preferences(self, query: str, k: int | None = None) -> list[dict]:
        """
        Semantic search across all user preferences with metadata filtering.
        Only returns actual user preference docs (not system docs).
        """
        store = self._load_or_create_store()
        k = k or settings.rag_top_k

        # Filter to only user preference docs
        docs = store.similarity_search(
            query,
            k=k,
            filter={"type": "preferences"},
        )

        results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        logger.debug("Semantic search '%s' returned %d docs", query, len(results))
        return results

    def update_preference(self, user_id: str, key: str, value: Any) -> None:
        """Update a single preference key for a user."""
        prefs = self.get_preferences(user_id)
        prefs[key] = value
        self.save_preferences(user_id, prefs)

    def add_brand_preference(self, user_id: str, brand: str, prefer: bool = True) -> None:
        """Convenience: add a brand to preferred or avoided list."""
        prefs = self.get_preferences(user_id)
        brand_lower = brand.lower()
        if prefer:
            brands = set(prefs.get("preferred_brands", []))
            brands.add(brand_lower)
            prefs["preferred_brands"] = sorted(brands)
            # Remove from avoid list if present (prefer wins over old avoid)
            prefs["avoid_brands"] = [b for b in prefs.get("avoid_brands", []) if b != brand_lower]
        else:
            brands = set(prefs.get("avoid_brands", []))
            brands.add(brand_lower)
            prefs["avoid_brands"] = sorted(brands)
            # Avoid always wins — remove from preferred
            prefs["preferred_brands"] = [b for b in prefs.get("preferred_brands", []) if b != brand_lower]
        self.save_preferences(user_id, prefs)

    def add_dietary_restriction(self, user_id: str, restriction: str) -> None:
        """Add a dietary restriction for a user."""
        prefs = self.get_preferences(user_id)
        restrictions = set(prefs.get("dietary_restrictions", []))
        restrictions.add(restriction.lower())
        prefs["dietary_restrictions"] = sorted(restrictions)
        self.save_preferences(user_id, prefs)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _default_preferences() -> dict[str, Any]:
    return {
        "preferred_brands": [],
        "avoid_brands": [],
        "dietary_restrictions": [],
        "quality_notes": "",
        "notes": "",
    }


def _resolve_preference_conflicts(prefs: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve contradictory preferences before saving.
    Rules (explicit override logic):
      - avoid_brands always wins over preferred_brands
      - vegan implies dairy-free
      - vegan implies vegetarian
    """
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
