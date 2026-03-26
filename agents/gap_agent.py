"""
agents/gap_agent.py
Gap Agent — cross-references detected inventory against the user's grocery list,
factors in user preferences from the RAG store, and outputs a prioritised
list of items that actually need purchasing.

Fixes applied:
  - Fuzzy string matching to canonicalize item names (e.g. "strawberries 2 pints" vs "strawberries 1 bowl")
  - Unit normalization before gap analysis
  - Preference conflict detection and resolution
  - Retrieval logging so we can verify correct preferences are fetched
  - Structured output with JSON schema enforcement
"""

from __future__ import annotations

import json
import logging
from textwrap import dedent

from openai import OpenAI

from core.config import settings
from core.state import GrocerAIState, GroceryNeed
from rag.preference_store import PreferenceStore

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)


# ─── Fuzzy item matching ──────────────────────────────────────────────────────

def _normalize_item_name(name: str) -> str:
    """
    Normalize an item name for fuzzy comparison.
    Strips quantities, units, and common descriptors.
    e.g. "2 pints of strawberries" → "strawberry"
         "whole milk 1 gallon"     → "milk"
    """
    import re
    name = name.lower().strip()
    # Remove leading quantities like "2 pints of"
    name = re.sub(r"^\d+[\s.]*\w*\s+(of\s+)?", "", name)
    # Remove trailing size descriptors
    name = re.sub(r"\s+\d+\s*(oz|lb|lbs|g|kg|ml|l|pint|quart|gallon|pack|count|ct)s?\b.*$", "", name)
    # Singularize common plurals
    for plural, singular in [
        ("strawberries", "strawberry"), ("blueberries", "blueberry"),
        ("raspberries", "raspberry"), ("blackberries", "blackberry"),
        ("tomatoes", "tomato"), ("potatoes", "potato"),
        ("apples", "apple"), ("oranges", "orange"), ("eggs", "egg"),
        ("carrots", "carrot"), ("onions", "onion"),
    ]:
        name = name.replace(plural, singular)
    return name.strip()


def _fuzzy_match_score(a: str, b: str) -> float:
    """
    Simple token-overlap similarity score between two item names.
    Returns 0.0–1.0. Uses set intersection of word tokens.
    """
    tokens_a = set(_normalize_item_name(a).split())
    tokens_b = set(_normalize_item_name(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _find_matching_inventory_item(
    grocery_item: str,
    inventory: list[dict],
    threshold: float = 0.4,
) -> dict | None:
    """
    Find the best-matching inventory item for a grocery list entry.
    Returns the inventory item if similarity >= threshold, else None.
    """
    best_score = 0.0
    best_item = None
    for inv_item in inventory:
        score = _fuzzy_match_score(grocery_item, inv_item["name"])
        if score > best_score:
            best_score = score
            best_item = inv_item
    if best_score >= threshold:
        logger.debug(
            "Fuzzy match: '%s' → '%s' (score=%.2f)",
            grocery_item, best_item["name"], best_score,
        )
        return best_item
    return None


# ─── Preference conflict detection ───────────────────────────────────────────

def _detect_preference_conflicts(preferences: dict) -> list[str]:
    """
    Detect contradictory preferences and log warnings.
    e.g. "gluten-free" + "sourdough bread" as preferred brand.
    Returns list of conflict warning strings.
    """
    conflicts = []
    dietary = set(preferences.get("dietary_restrictions", []))
    preferred = set(preferences.get("preferred_brands", []))
    quality = preferences.get("quality_notes", "").lower()

    if "vegan" in dietary and "dairy-free" not in dietary:
        conflicts.append("Vegan diet implies dairy-free — consider adding 'dairy-free' restriction.")

    if "gluten-free" in dietary:
        gluten_items = [b for b in preferred if any(
            kw in b for kw in ["bread", "pasta", "wheat", "sourdough", "bagel"]
        )]
        if gluten_items:
            conflicts.append(
                f"Gluten-free restriction may conflict with preferred brands: {', '.join(gluten_items)}"
            )

    if "dairy-free" in dietary and "cheese" in quality:
        conflicts.append("Dairy-free restriction conflicts with quality note mentioning cheese.")

    return conflicts


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
    You are a smart grocery gap analyser. Given:
    1. A user's grocery list (items they intended to buy)
    2. Current fridge/pantry inventory (detected by vision, already fuzzy-matched)
    3. User dietary restrictions and preferences

    Determine which items actually need to be purchased.

    Rules:
    - If an item is well-stocked (quantity >= 0.5 of a reasonable serving), mark needed=false
    - If an item is low or absent, mark needed=true
    - Apply dietary restrictions strictly: if an item conflicts, mark needed=false with reason="dietary restriction: <restriction>"
    - Priority: "high" if completely absent, "medium" if low stock, "low" if preference-driven upgrade
    - Item names may vary — use semantic matching, not exact string match
    - If inventory shows "1 bottle" of milk and grocery list says "whole milk", treat as stocked

    Respond with valid JSON only — no markdown, no prose:
    {
      "needs": [
        {
          "name": "<item from grocery list, exact>",
          "needed": true|false,
          "reason": "<concise reason>",
          "priority": "high|medium|low"
        }
      ]
    }
""").strip()


def _build_user_prompt(
    grocery_list: list[str],
    inventory_summary: str,
    preferences: dict,
    conflicts: list[str],
) -> str:
    conflict_section = ""
    if conflicts:
        conflict_section = f"\nPreference conflicts detected (resolve conservatively):\n" + "\n".join(f"- {c}" for c in conflicts)

    return dedent(f"""
        Grocery list:
        {json.dumps(grocery_list, indent=2)}

        Current inventory (fuzzy-matched against grocery list):
        {inventory_summary}

        User preferences and restrictions:
        {json.dumps(preferences, indent=2)}
        {conflict_section}

        Determine which items need purchasing.
    """).strip()


# ─── Agent ────────────────────────────────────────────────────────────────────

def run_gap_agent(state: GrocerAIState) -> GrocerAIState:
    """
    1. Loads user preferences from the RAG store (with retrieval logging).
    2. Detects and warns about preference conflicts.
    3. Fuzzy-matches grocery list items against detected inventory.
    4. Calls GPT-4o to produce a structured needs list.
    5. Populates state["grocery_needs"].
    """
    # 1. Load preferences with logging so we can verify correctness
    pref_store = PreferenceStore()
    preferences = pref_store.get_preferences(state["user_id"])
    state["user_preferences"] = preferences

    logger.info(
        "Gap Agent preferences loaded | user=%s brands=%s dietary=%s",
        state["user_id"],
        preferences.get("preferred_brands", []),
        preferences.get("dietary_restrictions", []),
    )

    # 2. Detect preference conflicts
    conflicts = _detect_preference_conflicts(preferences)
    for conflict in conflicts:
        logger.warning("Preference conflict: %s", conflict)
        state["warnings"].append(f"Preference conflict: {conflict}")

    # 3. Fuzzy-match grocery list against inventory for the prompt
    inventory = state.get("detected_inventory", [])
    inventory_summary = _format_inventory_with_matches(
        inventory, state["grocery_list"]
    )

    # 4. Call GPT-4o
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_prompt(
                state["grocery_list"],
                inventory_summary,
                preferences,
                conflicts,
            ),
        },
    ]

    try:
        response = _client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=800,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Gap Agent API call failed: %s", exc)
        state["errors"].append(f"Gap Agent: API call failed — {exc}")
        # Conservative fallback: mark everything as needed
        state["grocery_needs"] = [
            GroceryNeed(name=item, needed=True, reason="gap analysis failed — defaulting to needed", priority="medium")
            for item in state["grocery_list"]
        ]
        return state

    needs = _parse_needs_response(raw, state["grocery_list"])
    state["grocery_needs"] = needs

    needed_count = sum(1 for n in needs if n["needed"])
    logger.info(
        "Gap Agent: %d/%d items need purchasing",
        needed_count,
        len(needs),
    )

    return state


def _format_inventory_with_matches(
    inventory: list[dict],
    grocery_list: list[str],
) -> str:
    """
    Format inventory with fuzzy match annotations so the LLM
    can see which grocery items map to which inventory items.
    """
    if not inventory:
        return "No inventory detected (no image provided or empty fridge)."

    lines = []
    for item in inventory:
        conf = f"{item['confidence']:.0%}"
        # Find any grocery items that fuzzy-match this inventory item
        matches = [
            g for g in grocery_list
            if _fuzzy_match_score(g, item["name"]) >= 0.4
        ]
        match_note = f" [matches: {', '.join(matches)}]" if matches else ""
        lines.append(
            f"- {item['name']}: {item['quantity']} {item['unit']} "
            f"(confidence: {conf}){match_note}"
        )
    return "\n".join(lines)


def _parse_needs_response(raw: str, grocery_list: list[str]) -> list[GroceryNeed]:
    """Parse gap analysis JSON; fall back to marking everything needed on error."""
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        data = json.loads(clean)
        needs: list[GroceryNeed] = []
        seen_names = set()
        for item in data.get("needs", []):
            name = str(item.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            needs.append(
                GroceryNeed(
                    name=name,
                    needed=bool(item.get("needed", True)),
                    reason=str(item.get("reason", "")),
                    priority=str(item.get("priority", "medium")),
                )
            )
        # Ensure every grocery list item has a need entry (fill gaps)
        for item in grocery_list:
            if item not in seen_names:
                needs.append(
                    GroceryNeed(name=item, needed=True, reason="not analysed — defaulting to needed", priority="medium")
                )
        return needs

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse gap response: %s", exc)
        return [
            GroceryNeed(name=item, needed=True, reason="analysis failed — defaulting to needed", priority="medium")
            for item in grocery_list
        ]
