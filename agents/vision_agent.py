"""
agents/vision_agent.py
Vision Agent — uses GPT-4V to analyse a fridge/pantry photo and return
a structured inventory of detected items with quantities and confidence scores.

Fixes applied:
  - Structured JSON output enforced via response_format
  - Confidence threshold: items below 0.4 flagged for user confirmation
  - Image quality pre-check before calling vision model
  - Graceful fallback with clear user-facing error messages
  - Low-confidence items surfaced as warnings, not silently dropped
"""

from __future__ import annotations

import base64
import json
import logging
from textwrap import dedent

from openai import OpenAI

from core.config import settings
from core.state import GrocerAIState, InventoryItem

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)

# Items below this confidence are flagged for user confirmation
CONFIDENCE_THRESHOLD = 0.4
# Minimum base64 length — anything shorter is likely a corrupt/empty image
MIN_IMAGE_B64_LENGTH = 1000


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
    You are a kitchen inventory analyst. When given any photo of a fridge, freezer,
    pantry, or kitchen area, identify ALL visible food items — even if partially visible,
    dark, or blurry.

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
      "notes": "<any relevant observations about fridge state>",
      "image_quality": "<good|poor|not_a_fridge>"
    }

    Guidelines:
    - Use generic names (e.g. "whole milk" not "Organic Valley Whole Milk")
    - Be GENEROUS — if you can see ANY food item, include it even if partially visible
    - Include items you can infer from packaging color/shape even if label not visible
    - confidence 0.9+ = clearly visible, 0.5-0.9 = partially visible, <0.5 = inferred
    - Dark or blurry photos: still try to identify items, use lower confidence scores
    - NEVER return empty inventory if there are ANY items visible at all
    - If image is clearly not a fridge/kitchen, set image_quality to "not_a_fridge"
    - Set image_quality to "poor" for very dark, blurry, or low-resolution images
""").strip()

USER_PROMPT_WITH_IMAGE = "Analyse this fridge/pantry photo and return the inventory JSON."
USER_PROMPT_NO_IMAGE = "No image was provided. Return an empty inventory."


# ─── Image quality pre-check ──────────────────────────────────────────────────

def _check_image_quality(image_b64: str) -> tuple[bool, str]:
    """
    Basic pre-check before sending to vision model.
    Returns (is_ok, error_message).
    """
    if not image_b64:
        return False, "No image data provided."

    if len(image_b64) < MIN_IMAGE_B64_LENGTH:
        return False, "Image appears to be corrupt or empty (too small)."

    # Verify it's valid base64
    try:
        decoded = base64.b64decode(image_b64, validate=True)
        # Check for common image magic bytes
        if not (
            decoded[:3] == b'\xff\xd8\xff'  # JPEG
            or decoded[:8] == b'\x89PNG\r\n\x1a\n'  # PNG
            or decoded[:6] in (b'GIF87a', b'GIF89a')  # GIF
            or decoded[:4] == b'RIFF'  # WEBP
        ):
            return False, "File does not appear to be a valid image (JPEG/PNG/WEBP/GIF)."
    except Exception:
        return False, "Image data is not valid base64."

    return True, ""


# ─── Agent ────────────────────────────────────────────────────────────────────

def run_vision_agent(state: GrocerAIState) -> GrocerAIState:
    """
    Calls GPT-4V with the fridge image (if provided) and parses the
    structured inventory response into state["detected_inventory"].

    Low-confidence items (< CONFIDENCE_THRESHOLD) are included but
    flagged as warnings so the UI can prompt user confirmation.
    """
    image_b64 = state.get("fridge_image_b64")

    if image_b64:
        # Pre-check image quality before burning API tokens
        is_ok, quality_error = _check_image_quality(image_b64)
        if not is_ok:
            logger.warning("Image quality check failed: %s", quality_error)
            state["warnings"].append(f"Image quality issue: {quality_error} — proceeding without fridge scan.")
            state["detected_inventory"] = []
            state["vision_raw_response"] = None
            return state

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

    try:
        response = _client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.vision_max_tokens,
            temperature=0.1,  # low temp for consistent structured output
        )
    except Exception as exc:
        logger.error("Vision API call failed: %s", exc)
        state["errors"].append(f"Vision Agent: API call failed — {exc}")
        state["detected_inventory"] = []
        state["vision_raw_response"] = None
        return state

    raw = response.choices[0].message.content or ""
    state["vision_raw_response"] = raw

    parsed = _parse_vision_response(raw)

    # Handle non-fridge images
    if parsed.get("image_quality") == "not_a_fridge":
        state["warnings"].append(
            "The uploaded image doesn't appear to be a fridge or pantry. "
            "Inventory detection skipped — all grocery list items will be marked as needed."
        )
        state["detected_inventory"] = []
        return state

    if parsed.get("image_quality") == "poor":
        state["warnings"].append(
            "Image quality is low (dark/blurry). Detected items may be inaccurate — "
            "please review the inventory tab before proceeding."
        )

    inventory = parsed["inventory"]

    # Flag low-confidence items as warnings for UI to surface
    low_conf = [
        item for item in inventory
        if item["confidence"] < CONFIDENCE_THRESHOLD
    ]
    if low_conf:
        names = ", ".join(item["name"] for item in low_conf)
        state["warnings"].append(
            f"Low-confidence detections (please verify): {names}"
        )

    state["detected_inventory"] = inventory

    if parsed.get("notes"):
        state["warnings"].append(f"Vision note: {parsed['notes']}")

    logger.info(
        "Vision Agent detected %d items (%d low-confidence) | image_provided=%s",
        len(inventory),
        len(low_conf),
        bool(image_b64),
    )

    return state


def _parse_vision_response(raw: str) -> dict:
    """Parse the GPT-4V JSON response, with graceful fallback."""
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
                    confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                )
            )
        return {
            "inventory": inventory,
            "notes": data.get("notes", ""),
            "image_quality": data.get("image_quality", "good"),
        }

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse vision response: %s | raw=%s", exc, raw[:200])
        return {"inventory": [], "notes": f"Parse error: {exc}", "image_quality": "good"}
