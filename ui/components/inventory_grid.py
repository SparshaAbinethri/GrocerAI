"""
ui/components/inventory_grid.py
Reusable Streamlit component for displaying detected fridge inventory
as a colour-coded grid of metric cards.
"""

import sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from __future__ import annotations

import streamlit as st

from core.state import InventoryItem


# Confidence thresholds
HIGH_CONF = 0.80
MED_CONF = 0.50


def render_inventory_grid(inventory: list[InventoryItem], cols: int = 3) -> None:
    """
    Render detected inventory as a responsive grid of metric cards.
    Cards are colour-coded by confidence: green (high), yellow (medium), red (low).

    Args:
        inventory: List of InventoryItem dicts from the Vision Agent.
        cols: Number of columns in the grid (default 3).
    """
    if not inventory:
        st.info(
            "📷 No inventory detected. Upload a fridge photo on the previous step "
            "to automatically filter out items you already have."
        )
        return

    # Sort: high confidence first, then alphabetically
    sorted_inventory = sorted(
        inventory,
        key=lambda i: (-i["confidence"], i["name"]),
    )

    st.caption(
        f"Detected **{len(inventory)}** items — "
        f"{_count_by_confidence(inventory, HIGH_CONF)} high confidence · "
        f"{_count_by_confidence(inventory, MED_CONF, HIGH_CONF)} medium · "
        f"{_count_by_confidence(inventory, 0, MED_CONF)} low"
    )

    grid_cols = st.columns(cols)
    for i, item in enumerate(sorted_inventory):
        with grid_cols[i % cols]:
            _render_item_card(item)


def render_inventory_summary_text(inventory: list[InventoryItem]) -> str:
    """
    Return a human-readable summary string of detected inventory.
    Useful for debug panels or export.
    """
    if not inventory:
        return "No items detected."
    lines = [
        f"- {item['name'].title()}: {item['quantity']} {item['unit']} "
        f"({_conf_label(item['confidence'])} confidence)"
        for item in sorted(inventory, key=lambda x: x["name"])
    ]
    return "\n".join(lines)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _render_item_card(item: InventoryItem) -> None:
    conf = item["confidence"]
    icon = "🟢" if conf >= HIGH_CONF else "🟡" if conf >= MED_CONF else "🔴"
    label = f"{icon} {item['name'].title()}"
    value = f"{item['quantity']} {item['unit']}"
    delta = f"{conf:.0%} confidence"

    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color="normal" if conf >= MED_CONF else "inverse",
    )


def _conf_label(confidence: float) -> str:
    if confidence >= HIGH_CONF:
        return "high"
    elif confidence >= MED_CONF:
        return "medium"
    return "low"


def _count_by_confidence(
    inventory: list[InventoryItem],
    low: float = 0.0,
    high: float = 1.01,
) -> int:
    return sum(1 for i in inventory if low <= i["confidence"] < high)
