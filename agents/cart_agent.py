"""
agents/cart_agent.py
Cart Agent — assembles the final cart from Search Agent results,
handles user overrides, computes totals, and optionally submits
to Kroger's cart API. Always surfaces a review step before checkout.
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
    Does NOT auto-checkout — sets state["cart_items"] for UI review.
    Checkout is triggered explicitly by the user via the Streamlit UI.
    """
    search_results = state.get("search_results", {})
    grocery_needs = state.get("grocery_needs", [])

    cart_items: list[CartItem] = []

    for need in grocery_needs:
        if not need["needed"]:
            continue

        products = search_results.get(need["name"], [])

        if not products:
            logger.warning("No products found for '%s' — skipping", need["name"])
            state["warnings"].append(f"No products found for '{need['name']}'")
            continue

        best = products[0]  # already sorted by preference + unit price

        cart_items.append(
            CartItem(
                grocery_need=need["name"],
                selected_product=best,
                quantity=1,  # default; user can adjust in UI
                subtotal=round(best["price"] * 1, 2),
                override=False,
            )
        )

    state["cart_items"] = cart_items
    state["cart_total"] = round(sum(item["subtotal"] for item in cart_items), 2)

    logger.info(
        "Cart Agent: %d items | total=$%.2f",
        len(cart_items),
        state["cart_total"],
    )

    return state


def update_cart_item(
    cart_items: list[CartItem],
    need_name: str,
    new_product: ProductResult | None = None,
    new_quantity: int | None = None,
) -> list[CartItem]:
    """
    Apply a user override to a specific cart item (called from UI).
    Returns a new list with the updated item.
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
    Submit cart to Kroger API.
    Requires a valid Kroger user OAuth access token.

    Returns a dict with success, order_id, total, error.
    """
    kroger = KrogerClient()

    items_payload = [
        {
            "upc": item["selected_product"]["product_id"],
            "quantity": item["quantity"],
            "modality": "PICKUP",
        }
        for item in cart_items
        if item["selected_product"]["in_stock"]
    ]

    if not items_payload:
        return {
            "success": False,
            "order_id": None,
            "total": 0.0,
            "error": "No in-stock items to add to cart",
        }

    result = kroger.add_to_cart(items_payload, user_access_token)
    return result
