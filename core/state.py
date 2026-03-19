"""
core/state.py
Shared TypedDict state schema passed between all LangGraph agents.
Every agent reads from and writes to this single state object.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


# ─── Individual item models (plain dicts for LangGraph compatibility) ──────────

class InventoryItem(TypedDict):
    name: str               # e.g. "whole milk"
    quantity: float         # estimated amount (units vary by item)
    unit: str               # "carton", "oz", "lbs", etc.
    confidence: float       # 0.0–1.0 from vision model


class GroceryNeed(TypedDict):
    name: str               # original item from user's list
    needed: bool            # True if not sufficiently stocked
    reason: str             # e.g. "not detected in fridge"
    priority: str           # "high" | "medium" | "low"


class ProductResult(TypedDict, total=False):
    product_id: str
    name: str
    brand: str
    store: str              # "kroger" | "walmart"
    price: float
    unit_price: float       # price per unit/oz for comparison
    unit: str
    image_url: Optional[str]
    in_stock: bool
    preferred: bool         # True if matches RAG brand preference
    price_estimated: bool   # True if price is estimated (sandbox mode, no location ID)


class CartItem(TypedDict):
    grocery_need: str       # original list item name
    selected_product: ProductResult
    quantity: int
    subtotal: float
    override: bool          # True if user manually changed selection


class CheckoutResult(TypedDict):
    success: bool
    order_id: Optional[str]
    store: str
    total: float
    items_count: int
    error: Optional[str]


# ─── Top-level pipeline state ──────────────────────────────────────────────────

class GrocerAIState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────────
    session_id: str
    user_id: str
    fridge_image_b64: Optional[str]         # base64-encoded fridge photo
    grocery_list: list[str]                 # raw list from user input
    location_id: str                        # Kroger store location ID

    # ── Vision Agent output ──────────────────────────────────────────────────────
    detected_inventory: list[InventoryItem]
    vision_raw_response: Optional[str]      # raw GPT-4V response for debugging
    vision_error: Optional[str]

    # ── Gap Agent output ─────────────────────────────────────────────────────────
    grocery_needs: list[GroceryNeed]
    user_preferences: dict[str, Any]        # pulled from RAG store
    gap_error: Optional[str]

    # ── Search Agent output ──────────────────────────────────────────────────────
    search_results: dict[str, list[ProductResult]]  # need_name → ranked products
    price_summary: dict[str, float]         # store → estimated total
    search_error: Optional[str]

    # ── Cart Agent output ────────────────────────────────────────────────────────
    cart_items: list[CartItem]
    cart_total: float
    checkout_result: Optional[CheckoutResult]
    cart_error: Optional[str]

    # ── Pipeline metadata ────────────────────────────────────────────────────────
    current_step: str                       # tracks which agent is active
    errors: list[str]                       # accumulated error messages
    warnings: list[str]
    recommendations: list[dict]  # personalised product suggestions
    completed: bool
