"""
agents/search_agent.py
Search Agent — queries Kroger (and stub Walmart) APIs for each needed item,
applies brand preferences from the RAG store, and ranks results by
value (price-per-unit). Populates state["search_results"] and state["price_summary"].
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from api.kroger import KrogerClient
from core.config import settings
from core.state import GrocerAIState, GroceryNeed, ProductResult

# ── Rough price estimates for common grocery categories (USD) ─────────────────
# Used as fallback when Kroger sandbox returns $0.00 (no location ID set).
_PRICE_ESTIMATES: dict[str, float] = {
    "milk": 4.49, "whole milk": 4.49, "2% milk": 4.29, "skim milk": 3.99,
    "eggs": 3.99, "egg": 3.99,
    "butter": 5.49,
    "cheese": 5.99, "cheddar": 5.99, "mozzarella": 4.99, "parmesan": 6.99,
    "bread": 3.99, "sourdough": 5.49, "bagel": 4.99,
    "chicken": 7.99, "chicken breast": 8.99, "ground beef": 6.99,
    "pasta": 1.99, "spaghetti": 1.99, "penne": 1.99,
    "tomato sauce": 2.49, "marinara": 2.99,
    "olive oil": 8.99, "vegetable oil": 4.99,
    "rice": 3.49, "brown rice": 3.99,
    "yogurt": 1.49, "greek yogurt": 5.99,
    "orange juice": 4.99, "apple juice": 3.99,
    "spinach": 3.49, "lettuce": 2.99, "kale": 3.49,
    "apples": 4.99, "bananas": 1.29, "oranges": 4.49,
    "onions": 1.99, "garlic": 1.49, "potato": 3.99, "potatoes": 3.99,
    "carrots": 2.49, "broccoli": 2.99, "tomatoes": 3.49,
    "coffee": 9.99, "tea": 4.99,
    "cereal": 4.99, "oatmeal": 3.99,
    "peanut butter": 3.99, "jelly": 3.49,
    "sugar": 3.49, "flour": 3.99, "salt": 1.29,
    "soap": 2.99, "shampoo": 5.99,
}

def _estimate_price(item_name: str) -> float:
    """Return a rough price estimate for an item when live pricing unavailable."""
    name = item_name.lower().strip()
    # Exact match first
    if name in _PRICE_ESTIMATES:
        return _PRICE_ESTIMATES[name]
    # Partial match
    for key, price in _PRICE_ESTIMATES.items():
        if key in name or name in key:
            return price
    return 0.0  # truly unknown

logger = logging.getLogger(__name__)


# ─── Agent ────────────────────────────────────────────────────────────────────

def run_search_agent(state: GrocerAIState) -> GrocerAIState:
    """
    For each needed item, searches Kroger for matching products.
    Applies brand preference boosting from the RAG preference store.
    Results are sorted: preferred brands first, then by unit price.
    """
    needed_items = [n for n in state["grocery_needs"] if n["needed"]]

    if not needed_items:
        logger.info("No items needed — skipping search")
        state["search_results"] = {}
        state["price_summary"] = {}
        return state

    kroger = KrogerClient()
    preferences = state.get("user_preferences", {})
    brand_prefs: list[str] = [b.lower() for b in preferences.get("preferred_brands", [])]
    avoid_brands: list[str] = [b.lower() for b in preferences.get("avoid_brands", [])]
    location_id = state.get("location_id") or settings.kroger_location_id

    search_results: dict[str, list[ProductResult]] = {}
    kroger_total = 0.0

    # Run searches concurrently (one thread per item, capped at 5)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_item = {
            executor.submit(
                _search_item,
                kroger,
                need,
                location_id,
                brand_prefs,
                avoid_brands,
            ): need
            for need in needed_items
        }

        for future in as_completed(future_to_item):
            need = future_to_item[future]
            try:
                results = future.result()
                search_results[need["name"]] = results
                # Add cheapest matching product to store total
                if results:
                    kroger_total += results[0]["price"]  # uses estimated price if sandbox
            except Exception as exc:
                logger.warning("Search failed for '%s': %s", need["name"], exc)
                state["warnings"].append(f"Search: could not find '{need['name']}'")
                search_results[need["name"]] = []

    state["search_results"] = search_results
    state["price_summary"] = {"kroger": round(kroger_total, 2)}

    found = sum(1 for r in search_results.values() if r)
    logger.info(
        "Search Agent: found products for %d/%d items | kroger_est=$%.2f",
        found,
        len(needed_items),
        kroger_total,
    )

    return state


def _search_item(
    kroger: "KrogerClient",
    need: GroceryNeed,
    location_id: str,
    brand_prefs: list[str],
    avoid_brands: list[str],
) -> list[ProductResult]:
    """Search a single item across Kroger, apply preference ranking."""
    raw_results = kroger.search_products(
        query=need["name"],
        location_id=location_id,
        limit=settings.max_search_results_per_item,
    )

    products: list[ProductResult] = []
    for item in raw_results:
        brand = (item.get("brand") or "").lower()

        # Skip avoided brands entirely
        if any(avoid in brand for avoid in avoid_brands):
            continue

        preferred = any(pref in brand for pref in brand_prefs)

        # Extract price safely
        price_info = item.get("items", [{}])[0] if item.get("items") else {}
        price = float((price_info.get("price") or {}).get("regular", 0) or 0)

        # Extract size for unit price calculation
        size_str = price_info.get("size", "1")
        unit_price = _calc_unit_price(price, size_str)

        products.append(
            ProductResult(
                product_id=str(item.get("productId", "")),
                name=str(item.get("description", need["name"])),
                brand=str(item.get("brand", "Unknown")),
                store="kroger",
                price=price,
                unit_price=unit_price,
                unit=_extract_unit(size_str),
                image_url=_extract_image(item),
                in_stock=item.get("aisleLocations") is not None,
                preferred=preferred,
            )
        )

    # Apply price estimates for sandbox mode ($0.00 → estimated price)
    for p in products:
        if p["price"] == 0.0:
            estimated = _estimate_price(need["name"])
            if estimated > 0:
                p["price"] = estimated
                p["unit_price"] = _calc_unit_price(estimated, "16 oz")
                p["price_estimated"] = True  # flag so UI can show ~$x.xx

    # Sort: preferred first, then by unit price ascending
    products.sort(key=lambda p: (not p["preferred"], p["unit_price"]))
    return products


def _calc_unit_price(price: float, size_str: str) -> float:
    """Attempt to calculate price-per-oz or price-per-unit from size string."""
    if not price:
        return 0.0
    try:
        import re
        nums = re.findall(r"[\d.]+", str(size_str))
        if nums:
            return round(price / float(nums[0]), 4)
    except Exception:
        pass
    return price


def _extract_unit(size_str: str) -> str:
    import re
    units = re.findall(r"[a-zA-Z]+", str(size_str))
    return units[0].lower() if units else "unit"


def _extract_image(item: dict) -> str | None:
    images = item.get("images", [])
    if images:
        sizes = images[0].get("sizes", [])
        for size in sizes:
            if size.get("size") == "medium":
                return size.get("url")
        if sizes:
            return sizes[0].get("url")
    return None
