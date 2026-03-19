"""
agents/vision_agent.py
Vision Agent — uses GPT-4V to analyse a fridge/pantry photo and return
a structured inventory of detected items with quantities and confidence scores.
"""

from __future__ import annotations

import json
import logging
from textwrap import dedent

from openai import OpenAI

from core.config import settings
from core.state import GrocerAIState, InventoryItem

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
    You are a precise kitchen inventory analyst. When given a fridge or pantry photo,
    identify every visible food item with quantity estimates.

    Always respond with valid JSON only — no markdown fences, no prose.
    Schema:
    {
      "inventory": [
        {
          "name": "<canonical item name, lowercase>",
          "quantity": <numeric estimate>,
          "unit": "<carton|bottle|lbs|oz|bunch|bag|jar|can|piece|box>",
          "confidence": <0.0 to 1.0>
        }
      ],
      "notes": "<any relevant observations about fridge state>"
    }

    Guidelines:
    - Use generic names (e.g. "whole milk" not "Organic Valley Whole Milk")
    - Estimate quantity conservatively (half-full milk = 0.5 carton)
    - confidence 0.9+ = clearly visible, 0.5–0.9 = partially visible, <0.5 = inferred
    - If no image provided or image is not a fridge/pantry, return empty inventory with a note
""").strip()

USER_PROMPT_WITH_IMAGE = "Analyse this fridge/pantry photo and return the inventory JSON."
USER_PROMPT_NO_IMAGE = "No image was provided. Return an empty inventory."


# ─── Agent ────────────────────────────────────────────────────────────────────

def run_vision_agent(state: GrocerAIState) -> GrocerAIState:
    """
    Calls GPT-4V with the fridge image (if provided) and parses the
    structured inventory response into state["detected_inventory"].
    """
    image_b64 = state.get("fridge_image_b64")

    if image_b64:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": USER_PROMPT_WITH_IMAGE},
                ],
            },
        ]
    else:
        logger.info("No fridge image provided — returning empty inventory")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_NO_IMAGE},
        ]

    response = _client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=settings.vision_max_tokens,
        temperature=0.1,  # low temp for consistent structured output
    )

    raw = response.choices[0].message.content or ""
    state["vision_raw_response"] = raw

    parsed = _parse_vision_response(raw)
    state["detected_inventory"] = parsed["inventory"]

    if parsed.get("notes"):
        state["warnings"].append(f"Vision note: {parsed['notes']}")

    logger.info(
        "Vision Agent detected %d items (image_provided=%s)",
        len(state["detected_inventory"]),
        bool(image_b64),
    )

    return state


def _parse_vision_response(raw: str) -> dict:
    """Parse the GPT-4V JSON response, with graceful fallback."""
    # Strip accidental markdown fences if model disobeys instructions
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        data = json.loads(clean)
        inventory: list[InventoryItem] = []
        for item in data.get("inventory", []):
            inventory.append(
                InventoryItem(
                    name=str(item.get("name", "unknown")).lower().strip(),
                    quantity=float(item.get("quantity", 1.0)),
                    unit=str(item.get("unit", "piece")),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return {"inventory": inventory, "notes": data.get("notes", "")}

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse vision response: %s | raw=%s", exc, raw[:200])
        return {"inventory": [], "notes": f"Parse error: {exc}"}
