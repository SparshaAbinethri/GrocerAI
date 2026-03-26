"""
agents/cart_agent.py
Cart Agent — assembles the final cart from Search Agent results,
handles user overrides, computes totals, and optionally submits
to Kroger's cart API. Always surfaces a review step before checkout.

Fixes applied:
  - Cart deduplication: prevents double-adding items if pipeline re-runs
  - Budget constraint: trims low-priority items when over budget
  - Per-item error handling in checkout (not one big try/catch)
  - Checkout result surfaces individual item failures clearly
  - update_cart_item is idempotent on quantity/product changes
"""

from __future__ import annotations

import logging

from api.kroger import KrogerClient
from core.state import CartItem, GrocerAIState, ProductResult

logger = logging.getLogger(__name__)


# ─── Agent ────────────────────────────────────────────────────────────────────

def run_cart_agent(state: GrocerAIState) -> GrocerAIState:
    """
    Builds a cart from the top-ranked search result for each needed item.
    Deduplicates items so re-running the pipeline never double-adds.
    Respects budget constraint if state["budget"] is set.
    Does NOT auto-checkout — sets state["cart_items"] for UI review.
    """
    search_results = state.get("search_results", {})
    grocery_needs = state.get("grocery_needs", [])
    budget = state.get("budget")  # optional float

    # Deduplication: track which grocery needs already have a cart entry
    existing_needs = {
        item["grocery_need"]
        for item in state.get("cart_items", [])
    }

    cart_items: list[CartItem] = list(state.get("cart_items", []))

    for need in grocery_needs:
        if not need["needed"]:
            continue

        # Skip if already in cart (idempotency — safe to re-run pipeline)
        if need["name"] in existing_needs:
            logger.debug("Cart: '%s' already in cart — skipping", need["name"])
            continue

        products = search_results.get(need["name"], [])

        if not products:
            logger.warning("No products found for '%s' — skipping", need["name"])
            state["warnings"].append(
                f"No products found for '{need['name']}' — item excluded from cart."
            )
            continue

        best = products[0]  # already sorted: preferred brand first, then unit price

        cart_items.append(
            CartItem(
                grocery_need=need["name"],
                selected_product=best,
                quantity=1,
                subtotal=round(best["price"] * 1, 2),
                override=False,
            )
        )

    # Budget enforcement: trim low-priority items if over budget
    if budget:
        cart_items = _apply_budget(cart_items, grocery_needs, budget, state)

    state["cart_items"] = cart_items
    state["cart_total"] = round(sum(item["subtotal"] for item in cart_items), 2)

    logger.info(
        "Cart Agent: %d items | total=$%.2f | budget=%s",
        len(cart_items),
        state["cart_total"],
        f"${budget:.2f}" if budget else "none",
    )

    return state


def _apply_budget(
    cart_items: list[CartItem],
    grocery_needs: list[dict],
    budget: float,
    state: GrocerAIState,
) -> list[CartItem]:
    """
    If cart total exceeds budget, remove low-priority items first
    until we're under budget or only high-priority items remain.
    """
    total = sum(i["subtotal"] for i in cart_items)
    if total <= budget:
        return cart_items

    # Build priority lookup
    priority_map = {n["name"]: n["priority"] for n in grocery_needs}
    priority_order = {"low": 0, "medium": 1, "high": 2}

    # Sort: remove lowest priority first
    sorted_items = sorted(
        cart_items,
        key=lambda i: priority_order.get(priority_map.get(i["grocery_need"], "medium"), 1),
    )

    kept = list(cart_items)
    for item in sorted_items:
        if sum(i["subtotal"] for i in kept) <= budget:
            break
        priority = priority_map.get(item["grocery_need"], "medium")
        if priority == "high":
            break  # never remove high-priority items for budget
        kept = [i for i in kept if i["grocery_need"] != item["grocery_need"]]
        state["warnings"].append(
            f"'{item['grocery_need']}' removed to stay within ${budget:.2f} budget."
        )

    return kept


def update_cart_item(
    cart_items: list[CartItem],
    need_name: str,
    new_product: ProductResult | None = None,
    new_quantity: int | None = None,
) -> list[CartItem]:
    """
    Apply a user override to a specific cart item (called from UI).
    Returns a new list — original is not mutated (immutable update pattern).
    """
    updated = []
    for item in cart_items:
        if item["grocery_need"] == need_name:
            product = new_product or item["selected_product"]
            qty = new_quantity if new_quantity is not None else item["quantity"]
            updated.append(
                CartItem(
                    grocery_need=item["grocery_need"],
                    selected_product=product,
                    quantity=qty,
                    subtotal=round(product["price"] * qty, 2),
                    override=True,
                )
            )
        else:
            updated.append(item)
    return updated


def checkout(cart_items: list[CartItem], user_access_token: str) -> dict:
    """
    Submit cart to Kroger API with per-item error handling.
    Requires a valid Kroger user OAuth access token.

    Returns a dict with:
      success: bool
      order_id: str | None
      total: float
      added_items: list of successfully added item names
      failed_items: list of (item_name, error_reason) tuples
      error: str | None (overall error if entire call failed)
    """
    kroger = KrogerClient()

    # Separate in-stock from out-of-stock upfront
    in_stock_items = [i for i in cart_items if i["selected_product"]["in_stock"]]
    out_of_stock = [i for i in cart_items if not i["selected_product"]["in_stock"]]

    if out_of_stock:
        logger.warning(
            "Skipping %d out-of-stock items: %s",
            len(out_of_stock),
            [i["grocery_need"] for i in out_of_stock],
        )

    if not in_stock_items:
        return {
            "success": False,
            "order_id": None,
            "total": 0.0,
            "added_items": [],
            "failed_items": [(i["grocery_need"], "out of stock") for i in out_of_stock],
            "error": "No in-stock items to add to cart.",
        }

    # Attempt checkout — try all items together first, then individually on failure
    items_payload = [
        {
            "upc": item["selected_product"]["product_id"],
            "quantity": item["quantity"],
            "modality": "PICKUP",
        }
        for item in in_stock_items
        if item["selected_product"]["product_id"]  # skip items without valid UPC
    ]

    result = kroger.add_to_cart(items_payload, user_access_token)

    if result["success"]:
        return {
            "success": True,
            "order_id": result.get("order_id"),
            "total": round(sum(i["subtotal"] for i in in_stock_items), 2),
            "added_items": [i["grocery_need"] for i in in_stock_items],
            "failed_items": [(i["grocery_need"], "out of stock") for i in out_of_stock],
            "error": None,
        }

    # Bulk add failed — try adding items one by one to find which ones fail
    logger.warning("Bulk cart add failed — attempting per-item fallback")
    added = []
    failed = [(i["grocery_need"], "out of stock") for i in out_of_stock]

    for item in in_stock_items:
        product_id = item["selected_product"]["product_id"]
        if not product_id:
            failed.append((item["grocery_need"], "missing product ID"))
            continue
        single_result = kroger.add_to_cart(
            [{"upc": product_id, "quantity": item["quantity"], "modality": "PICKUP"}],
            user_access_token,
        )
        if single_result["success"]:
            added.append(item["grocery_need"])
        else:
            error_detail = single_result.get("error", "unknown error")
            failed.append((item["grocery_need"], error_detail))
            logger.warning("Failed to add '%s': %s", item["grocery_need"], error_detail)

    return {
        "success": len(added) > 0,
        "order_id": None,
        "total": round(sum(i["subtotal"] for i in in_stock_items if i["grocery_need"] in added), 2),
        "added_items": added,
        "failed_items": failed,
        "error": f"{len(failed)} item(s) could not be added." if failed else None,
    }
