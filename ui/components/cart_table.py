"""
ui/components/cart_table.py
Reusable Streamlit component for rendering and editing the cart.
Extracted from app.py so it can be tested and reused independently.
"""

import sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from __future__ import annotations

import streamlit as st

from agents.cart_agent import update_cart_item
from core.state import CartItem, ProductResult


def render_cart_table(
    cart_items: list[CartItem],
    search_results: dict[str, list[ProductResult]],
    session_key: str = "cart_items",
) -> tuple[list[CartItem], bool]:
    """
    Render the cart as an interactive table with:
      - Product image, name, brand, unit price
      - Quantity stepper
      - Alternative product selector (from search results)
      - Preferred brand badge

    Args:
        cart_items: Current cart items.
        search_results: All search results keyed by grocery need name.
        session_key: Streamlit session_state key to update on changes.

    Returns:
        (updated_cart_items, changed) — changed=True if user modified anything.
    """
    updated_items = list(cart_items)
    changed = False

    for idx, cart_item in enumerate(cart_items):
        need_name = cart_item["grocery_need"]
        product = cart_item["selected_product"]
        alternatives = search_results.get(need_name, [])

        with st.expander(
            _expander_label(cart_item),
            expanded=False,
        ):
            img_col, info_col, qty_col, swap_col = st.columns([1, 3, 1, 2])

            with img_col:
                if product.get("image_url"):
                    st.image(product["image_url"], width=70)
                else:
                    st.markdown("🛒")

            with info_col:
                st.markdown(f"**{product['name']}**")
                st.caption(f"{product['brand']} · {product['store'].title()}")
                st.caption(f"${product['price']:.2f} · ${product['unit_price']:.3f}/{product['unit']}")
                _render_badges(product)

            with qty_col:
                new_qty = st.number_input(
                    "Qty",
                    min_value=1,
                    max_value=20,
                    value=cart_item["quantity"],
                    key=f"qty_{session_key}_{idx}",
                    label_visibility="collapsed",
                )
                st.caption("qty")
                if new_qty != cart_item["quantity"]:
                    updated_items = update_cart_item(updated_items, need_name, new_quantity=new_qty)
                    changed = True

            with swap_col:
                if len(alternatives) > 1:
                    alt_labels = [
                        f"{'⭐ ' if a['preferred'] else ''}{a['brand']} · ${a['price']:.2f}"
                        for a in alternatives[:5]
                    ]
                    current_idx = next(
                        (i for i, a in enumerate(alternatives[:5])
                         if a["product_id"] == product["product_id"]),
                        0,
                    )
                    selected_label = st.selectbox(
                        "Switch",
                        options=alt_labels,
                        index=current_idx,
                        key=f"alt_{session_key}_{idx}",
                    )
                    selected_alt_idx = alt_labels.index(selected_label)
                    if selected_alt_idx != current_idx:
                        updated_items = update_cart_item(
                            updated_items,
                            need_name,
                            new_product=alternatives[selected_alt_idx],
                        )
                        changed = True
                else:
                    st.caption("No alternatives")

    return updated_items, changed


def render_cart_summary(cart_items: list[CartItem]) -> None:
    """Render a compact price summary row below the cart table."""
    if not cart_items:
        return

    total = round(sum(i["subtotal"] for i in cart_items), 2)
    item_count = len(cart_items)
    overrides = sum(1 for i in cart_items if i["override"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Items", item_count)
    c2.metric("Cart Total", f"${total:.2f}")
    if overrides:
        c3.metric("Customised", f"{overrides} item{'s' if overrides > 1 else ''}")


def _expander_label(cart_item: CartItem) -> str:
    """Build the expander header label for a cart item."""
    product = cart_item["selected_product"]
    preferred_star = "⭐ " if product["preferred"] else ""
    override_tag = " ✏️" if cart_item["override"] else ""
    return (
        f"{preferred_star}**{cart_item['grocery_need'].title()}** — "
        f"{product['brand']} · ${product['price']:.2f}{override_tag}"
    )


def _render_badges(product: ProductResult) -> None:
    """Render inline status badges for a product."""
    badges = []
    if product.get("preferred"):
        badges.append("⭐ Preferred brand")
    if not product.get("in_stock"):
        badges.append("⚠️ May be out of stock")
    if product.get("store") == "kroger":
        badges.append("🏪 Kroger")
    for badge in badges:
        st.caption(badge)
