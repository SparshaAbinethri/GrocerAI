"""
agents/search_agent.py
Search Agent — queries Kroger API for each needed item,
applies brand preferences from the RAG store, and ranks results by
value (price-per-unit). Populates state["search_results"] and state["price_summary"].

Fixes applied:
  - Exponential backoff + retry on rate limit / transient errors
  - Clear result ranking strategy (preferred brand > unit price)
  - Missing items explicitly surfaced to user (not silently dropped)
  - Substitute suggestions when item unavailable
  - Removed duplicate/noisy debug logging
  - Budget constraint support via state["budget"]
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from api.kroger import KrogerClient
from core.config import settings
from core.state import GrocerAIState, GroceryNeed, ProductResult

logger = logging.getLogger(__name__)

# ── Price estimates for sandbox mode (no location ID) ─────────────────────────
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
}

# ── Substitute map for common unavailable items ───────────────────────────────
_SUBSTITUTES: dict[str, list[str]] = {
    "whole milk": ["2% milk", "oat milk", "almond milk"],
    "sourdough bread": ["white bread", "whole wheat bread"],
    "chicken breast": ["chicken thighs", "turkey breast"],
    "ground beef": ["ground turkey", "ground chicken"],
    "butter": ["margarine", "coconut oil"],
    "parmesan": ["pecorino romano", "grana padano"],
}


def _estimate_price(item_name: str) -> float:
    name = item_name.lower().strip()
    if name in _PRICE_ESTIMATES:
        return _PRICE_ESTIMATES[name]
    for key, price in _PRICE_ESTIMATES.items():
        if key in name or name in key:
            return price
    return 0.0


def _get_substitutes(item_name: str) -> list[str]:
    name = item_name.lower().strip()
    for key, subs in _SUBSTITUTES.items():
        if key in name or name in key:
            return subs
    return []


# ─── Retry helper ─────────────────────────────────────────────────────────────

def _with_retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    """
    Call fn() with exponential backoff on transient errors.
    Raises the last exception if all retries exhausted.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            # Don't retry on auth errors — they won't resolve with retries
            if hasattr(exc, "response") and getattr(exc.response, "status_code", 0) in (401, 403):
                raise
            wait = base_delay * (2 ** attempt)
            logger.warning(
                "Search attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise last_exc


# ─── Agent ────────────────────────────────────────────────────────────────────

def run_search_agent(state: GrocerAIState) -> GrocerAIState:
    """
    For each needed item, searches Kroger for matching products.
    Applies brand preference boosting from the RAG preference store.
    Results are sorted: preferred brands first, then by unit price.
    Missing items are surfaced to the user with substitute suggestions.
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
    budget = state.get("budget")  # optional budget constraint

    search_results: dict[str, list[ProductResult]] = {}
    kroger_total = 0.0
    not_found_items: list[str] = []

    # Run searches concurrently (capped at 5 threads)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_item = {
            executor.submit(
                _search_item_with_retry,
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

                if results:
                    kroger_total += results[0]["price"]
                else:
                    # Item not found — surface to user with substitutes
                    not_found_items.append(need["name"])
                    subs = _get_substitutes(need["name"])
                    if subs:
                        state["warnings"].append(
                            f"'{need['name']}' not found on Kroger. "
                            f"Consider: {', '.join(subs)}"
                        )
                    else:
                        state["warnings"].append(
                            f"'{need['name']}' not found on Kroger — try searching manually."
                        )

            except Exception as exc:
                logger.warning("Search failed for '%s': %s", need["name"], exc)
                state["warnings"].append(
                    f"Could not search for '{need['name']}' — API error. Try again later."
                )
                search_results[need["name"]] = []

    # Budget enforcement: warn if over budget
    if budget and kroger_total > budget:
        over = kroger_total - budget
        state["warnings"].append(
            f"Estimated total ${kroger_total:.2f} exceeds your budget of ${budget:.2f} "
            f"(${over:.2f} over). Consider removing low-priority items."
        )

    state["search_results"] = search_results
    state["price_summary"] = {"kroger": round(kroger_total, 2)}

    found = sum(1 for r in search_results.values() if r)
    logger.info(
        "Search Agent: found %d/%d items | not_found=%s | kroger_est=$%.2f",
        found,
        len(needed_items),
        not_found_items,
        kroger_total,
    )

    return state


def _search_item_with_retry(
    kroger: KrogerClient,
    need: GroceryNeed,
    location_id: str,
    brand_prefs: list[str],
    avoid_brands: list[str],
) -> list[ProductResult]:
    """Search a single item with retry logic."""
    return _with_retry(
        lambda: _search_item(kroger, need, location_id, brand_prefs, avoid_brands),
        max_retries=3,
        base_delay=1.0,
    )


def _search_item(
    kroger: KrogerClient,
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

        # Extract price safely from Kroger response structure
        price_info = item.get("items", [{}])[0] if item.get("items") else {}
        price_obj = price_info.get("price") or {}
        price = float(
            price_obj.get("regular")
            or price_obj.get("promo")
            or price_obj.get("regularPerUnitEstimate")
            or item.get("regularPrice")
            or item.get("price")
            or 0
        )

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
                price_estimated=False,
            )
        )

    # Apply price estimates for sandbox mode ($0.00 → estimated price)
    for p in products:
        if p["price"] == 0.0:
            estimated = _estimate_price(need["name"])
            if estimated > 0:
                p["price"] = estimated
                p["unit_price"] = _calc_unit_price(estimated, "16 oz")
                p["price_estimated"] = True

    # Ranking strategy: preferred brand first, then unit price ascending
    # This is explicit and documented — not random
    products.sort(key=lambda p: (not p["preferred"], p["unit_price"]))
    return products


def _calc_unit_price(price: float, size_str: str) -> float:
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
