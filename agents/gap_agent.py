"""
agents/gap_agent.py
Gap Agent — cross-references detected inventory against the user's grocery list,
factors in user preferences from the RAG store, and outputs a prioritised
list of items that actually need purchasing.
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


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
    You are a smart grocery gap analyser. Given:
    1. A user's grocery list (items they intended to buy)
    2. Current fridge/pantry inventory (detected by vision)
    3. User dietary restrictions and preferences

    Determine which items actually need to be purchased.

    Rules:
    - If an item is well-stocked (≥0.5 units remaining), mark needed=false
    - If an item is low or absent, mark needed=true
    - Apply dietary restrictions: if an item conflicts, mark needed=false with reason="dietary restriction"
    - Priority: "high" if item is completely absent, "medium" if low, "low" if preference-driven upgrade
    - Normalise item names (e.g. "2% milk" and "milk" should match "whole milk" in inventory)

    Respond with valid JSON only:
    {
      "needs": [
        {
          "name": "<item from grocery list>",
          "needed": true|false,
          "reason": "<why needed or not>",
          "priority": "high|medium|low"
        }
      ]
    }
""").strip()


def _build_user_prompt(
    grocery_list: list[str],
    inventory_summary: str,
    preferences: dict,
) -> str:
    return dedent(f"""
        Grocery list:
        {json.dumps(grocery_list, indent=2)}

        Current inventory:
        {inventory_summary}

        User preferences and restrictions:
        {json.dumps(preferences, indent=2)}

        Determine which items need purchasing.
    """).strip()


# ─── Agent ────────────────────────────────────────────────────────────────────

def run_gap_agent(state: GrocerAIState) -> GrocerAIState:
    """
    1. Loads user preferences from the RAG store.
    2. Calls GPT-4 to cross-reference grocery list vs inventory.
    3. Populates state["grocery_needs"].
    """
    # 1. Load user preferences from RAG
    pref_store = PreferenceStore()
    preferences = pref_store.get_preferences(state["user_id"])
    state["user_preferences"] = preferences

    # 2. Build a readable inventory summary for the prompt
    inventory_summary = _format_inventory(state["detected_inventory"])

    # 3. Call GPT-4 (no vision needed here, text only is fine)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_prompt(
                state["grocery_list"],
                inventory_summary,
                preferences,
            ),
        },
    ]

    response = _client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=800,
        temperature=0.0,
    )

    raw = response.choices[0].message.content or ""
    needs = _parse_needs_response(raw, state["grocery_list"])
    state["grocery_needs"] = needs

    needed_count = sum(1 for n in needs if n["needed"])
    logger.info(
        "Gap Agent: %d/%d items need purchasing",
        needed_count,
        len(needs),
    )

    return state


def _format_inventory(inventory: list) -> str:
    if not inventory:
        return "No inventory detected (no image provided or empty fridge)."
    lines = []
    for item in inventory:
        conf = f"{item['confidence']:.0%}"
        lines.append(
            f"- {item['name']}: {item['quantity']} {item['unit']} (confidence: {conf})"
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
        for item in data.get("needs", []):
            needs.append(
                GroceryNeed(
                    name=str(item.get("name", "")),
                    needed=bool(item.get("needed", True)),
                    reason=str(item.get("reason", "")),
                    priority=str(item.get("priority", "medium")),
                )
            )
        return needs

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse gap response: %s", exc)
        # Conservative fallback: mark everything as needed
        return [
            GroceryNeed(name=item, needed=True, reason="analysis failed — defaulting to needed", priority="medium")
            for item in grocery_list
        ]
