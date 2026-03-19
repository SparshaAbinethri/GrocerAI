"""
agents/recommendations_agent.py
Generates personalised product recommendations based on:
  - User's dietary preferences and brand preferences from RAG store
  - What they already bought (cart items)
  - What was detected in the fridge
Uses GPT-4o to reason about gaps and suggest complementary items.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from core.config import settings
from core.state import GrocerAIState
from rag.preference_store import PreferenceStore

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def run_recommendations_agent(state: GrocerAIState) -> GrocerAIState:
    """
    Generate personalised product recommendations.
    Adds state["recommendations"] — list of recommendation dicts.
    """
    try:
        prefs = PreferenceStore().get_preferences(state["user_id"])
        cart_items = state.get("cart_items", [])
        inventory = state.get("detected_inventory", [])

        cart_names = [item["grocery_need"] for item in cart_items]
        fridge_names = [item["name"] for item in inventory]

        dietary = prefs.get("dietary_restrictions", [])
        preferred_brands = prefs.get("preferred_brands", [])
        quality_notes = prefs.get("quality_notes", "")

        prompt = f"""You are a helpful grocery shopping assistant.

Based on what the user is buying and what they already have, suggest 6 complementary products they might enjoy.

What they're buying: {", ".join(cart_names) if cart_names else "various groceries"}
Already in fridge: {", ".join(fridge_names) if fridge_names else "unknown"}
Dietary restrictions: {", ".join(dietary) if dietary else "none"}
Preferred brands: {", ".join(preferred_brands) if preferred_brands else "none specified"}
Quality preferences: {quality_notes or "none"}

Rules:
- Don't recommend things already in their cart or fridge
- Respect dietary restrictions strictly
- Suggest things that pair well or complement what they're buying
- Include a mix of: meal enhancers, snacks, staples they might be running low on
- If they have preferred brands, suggest those brands when relevant

Respond ONLY with a valid JSON array, no markdown, no explanation:
[
  {{
    "name": "product name",
    "reason": "one sentence why this pairs well or is useful",
    "category": "produce|dairy|meat|bakery|pantry|snacks|beverages|frozen",
    "estimated_price": 3.99,
    "kroger_search_url": "https://www.kroger.com/search?query=product+name"
  }}
]"""

        response = _get_client().chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        recommendations = json.loads(raw)

        # Ensure kroger_search_url is set
        for rec in recommendations:
            if not rec.get("kroger_search_url"):
                slug = rec["name"].replace(" ", "+")
                rec["kroger_search_url"] = f"https://www.kroger.com/search?query={slug}"

        state["recommendations"] = recommendations
        logger.info("Generated %d recommendations", len(recommendations))

    except Exception as e:
        logger.warning("Recommendations agent failed: %s", e)
        state["recommendations"] = []

    return state
