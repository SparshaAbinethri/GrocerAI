"""
ui/components/preference_panel.py
Reusable Streamlit component for the user preference panel.
Encapsulates all preference CRUD so it can be embedded in the sidebar
or a dedicated settings page.
"""

import sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from __future__ import annotations

import streamlit as st

from rag.preference_store import PreferenceStore

# Canonical dietary restriction options shown in the UI
DIETARY_OPTIONS = [
    "vegan",
    "vegetarian",
    "gluten-free",
    "dairy-free",
    "nut-free",
    "kosher",
    "halal",
    "low-sodium",
    "low-sugar",
    "keto",
    "paleo",
]


def render_preference_panel(user_id: str, compact: bool = True) -> dict:
    """
    Render the preference editing panel and return the current preferences dict.

    Args:
        user_id: The user whose preferences to load and save.
        compact: If True, renders in sidebar-friendly compact mode.
                 If False, renders as a full-width settings page.

    Returns:
        Current preferences dict (after any saves).
    """
    pref_store = PreferenceStore()
    prefs = pref_store.get_preferences(user_id)

    if compact:
        return _render_compact(user_id, prefs, pref_store)
    else:
        return _render_full(user_id, prefs, pref_store)


def _render_compact(user_id: str, prefs: dict, store: PreferenceStore) -> dict:
    """Sidebar-width compact layout."""
    st.subheader("🏷️ My Preferences")

    preferred_brands_text = st.text_area(
        "Preferred brands",
        value="\n".join(prefs.get("preferred_brands", [])),
        height=70,
        help="One brand per line. These products will be shown first.",
        key=f"pref_brands_{user_id}",
    )

    avoid_brands_text = st.text_area(
        "Avoid brands",
        value="\n".join(prefs.get("avoid_brands", [])),
        height=70,
        help="These brands will be hidden from results.",
        key=f"avoid_brands_{user_id}",
    )

    dietary = st.multiselect(
        "Dietary restrictions",
        options=DIETARY_OPTIONS,
        default=[r for r in prefs.get("dietary_restrictions", []) if r in DIETARY_OPTIONS],
        key=f"dietary_{user_id}",
    )

    quality_notes = st.text_input(
        "Quality notes",
        value=prefs.get("quality_notes", ""),
        placeholder="e.g. organic produce, grass-fed",
        key=f"quality_{user_id}",
    )

    if st.button("💾 Save", use_container_width=True, key=f"save_prefs_{user_id}"):
        updated = _build_prefs(preferred_brands_text, avoid_brands_text, dietary, quality_notes)
        store.save_preferences(user_id, updated)
        st.success("Saved!")
        return updated

    return prefs


def _render_full(user_id: str, prefs: dict, store: PreferenceStore) -> dict:
    """Full-width settings page layout with more detail."""
    st.header("⚙️ Shopping Preferences")
    st.caption(f"Preferences for session `{user_id}` — saved across sessions via FAISS.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Brands")
        preferred_brands_text = st.text_area(
            "✅ Preferred brands (one per line)",
            value="\n".join(prefs.get("preferred_brands", [])),
            height=120,
            help="These brands will be boosted to the top of search results.",
        )
        avoid_brands_text = st.text_area(
            "❌ Avoid these brands (one per line)",
            value="\n".join(prefs.get("avoid_brands", [])),
            height=120,
            help="Products from these brands will be completely excluded.",
        )

    with col2:
        st.subheader("Dietary & Quality")
        dietary = st.multiselect(
            "Dietary restrictions",
            options=DIETARY_OPTIONS,
            default=[r for r in prefs.get("dietary_restrictions", []) if r in DIETARY_OPTIONS],
        )
        quality_notes = st.text_area(
            "Quality preferences",
            value=prefs.get("quality_notes", ""),
            placeholder="e.g. organic produce, grass-fed beef, free-range eggs",
            height=80,
        )
        extra_notes = st.text_area(
            "Other notes",
            value=prefs.get("notes", ""),
            placeholder="Any other shopping preferences...",
            height=80,
        )

    st.divider()
    if st.button("💾 Save Preferences", type="primary"):
        updated = _build_prefs(preferred_brands_text, avoid_brands_text, dietary, quality_notes)
        updated["notes"] = extra_notes
        store.save_preferences(user_id, updated)
        st.success("✅ Preferences saved successfully!")
        st.balloons()
        return updated

    return prefs


def _build_prefs(
    preferred_text: str,
    avoid_text: str,
    dietary: list[str],
    quality_notes: str,
) -> dict:
    return {
        "preferred_brands": [b.strip().lower() for b in preferred_text.splitlines() if b.strip()],
        "avoid_brands": [b.strip().lower() for b in avoid_text.splitlines() if b.strip()],
        "dietary_restrictions": dietary,
        "quality_notes": quality_notes.strip(),
        "notes": "",
    }
