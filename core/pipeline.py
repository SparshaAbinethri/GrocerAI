"""
core/pipeline.py
LangGraph graph definition — wires Vision → Gap → Search → Cart agents.
Each node is a pure function that receives and returns GrocerAIState.

Fixes applied:
  - Error status propagated between nodes (downstream agents see upstream errors)
  - Budget field added to initial state
  - State serialization helper for session persistence
  - Pipeline rebuild cached (avoid recompiling on every request)
  - Cleaner finally block that handles missing final_state safely
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from agents.cart_agent import run_cart_agent
from agents.gap_agent import run_gap_agent
from agents.recommendations_agent import run_recommendations_agent
from agents.search_agent import run_search_agent
from agents.vision_agent import run_vision_agent
from core.db import log_pipeline_run
from core.metrics import record_pipeline_run
from core.state import GrocerAIState

logger = logging.getLogger(__name__)

# Cache the compiled pipeline — no need to recompile on every request
_compiled_pipeline = None


# ─── Node wrappers (error isolation + status propagation) ────────────────────

def vision_node(state: GrocerAIState) -> GrocerAIState:
    logger.info("[Pipeline] Running Vision Agent")
    state["current_step"] = "vision"
    try:
        return run_vision_agent(state)
    except Exception as exc:
        logger.exception("Vision Agent failed")
        state["vision_error"] = str(exc)
        state["errors"].append(f"Vision: {exc}")
        return state


def gap_node(state: GrocerAIState) -> GrocerAIState:
    logger.info("[Pipeline] Running Gap Agent")
    state["current_step"] = "gap"
    # Propagate vision error context so Gap Agent can adjust behavior
    if state.get("vision_error"):
        logger.warning("Gap Agent running with vision error: %s", state["vision_error"])
    try:
        return run_gap_agent(state)
    except Exception as exc:
        logger.exception("Gap Agent failed")
        state["gap_error"] = str(exc)
        state["errors"].append(f"Gap: {exc}")
        # Conservative fallback: mark everything as needed
        from core.state import GroceryNeed
        state["grocery_needs"] = [
            GroceryNeed(name=item, needed=True, reason="gap agent failed", priority="medium")
            for item in state.get("grocery_list", [])
        ]
        return state


def search_node(state: GrocerAIState) -> GrocerAIState:
    logger.info("[Pipeline] Running Search Agent")
    state["current_step"] = "search"
    if state.get("gap_error"):
        logger.warning("Search Agent running with gap error: %s", state["gap_error"])
    try:
        return run_search_agent(state)
    except Exception as exc:
        logger.exception("Search Agent failed")
        state["search_error"] = str(exc)
        state["errors"].append(f"Search: {exc}")
        return state


def cart_node(state: GrocerAIState) -> GrocerAIState:
    logger.info("[Pipeline] Running Cart Agent")
    state["current_step"] = "cart"
    if state.get("search_error"):
        logger.warning("Cart Agent running with search error: %s", state["search_error"])
    try:
        return run_cart_agent(state)
    except Exception as exc:
        logger.exception("Cart Agent failed")
        state["cart_error"] = str(exc)
        state["errors"].append(f"Cart: {exc}")
        return state


# ─── Conditional edges ────────────────────────────────────────────────────────

def should_continue_after_vision(state: GrocerAIState) -> str:
    """Skip to END if vision failed AND no grocery list to fall back on."""
    if state.get("vision_error") and not state.get("detected_inventory"):
        if not state.get("grocery_list"):
            logger.warning("Vision failed with no inventory and no grocery list — aborting")
            return "abort"
        # If we have a grocery list, continue — Gap Agent can work without inventory
        logger.info("Vision failed but grocery list present — continuing to Gap Agent")
    return "continue"


def should_continue_after_gap(state: GrocerAIState) -> str:
    """Skip search/cart if nothing is needed."""
    needs = state.get("grocery_needs", [])
    if not needs:
        logger.warning("Gap Agent produced no needs — aborting")
        return "abort"
    if not any(n["needed"] for n in needs):
        logger.info("All items already stocked — skipping Search & Cart")
        return "fully_stocked"
    return "continue"


# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_pipeline() -> Any:
    """Compile and return the LangGraph pipeline (cached after first build)."""
    global _compiled_pipeline
    if _compiled_pipeline is not None:
        return _compiled_pipeline

    graph = StateGraph(GrocerAIState)

    graph.add_node("vision", vision_node)
    graph.add_node("gap", gap_node)
    graph.add_node("search", search_node)
    graph.add_node("cart", cart_node)
    graph.add_node("recommend", run_recommendations_agent)

    graph.set_entry_point("vision")

    graph.add_conditional_edges(
        "vision",
        should_continue_after_vision,
        {"continue": "gap", "abort": END},
    )

    graph.add_conditional_edges(
        "gap",
        should_continue_after_gap,
        {"continue": "search", "fully_stocked": END, "abort": END},
    )

    graph.add_edge("search", "cart")
    graph.add_edge("cart", "recommend")
    graph.add_edge("recommend", END)

    _compiled_pipeline = graph.compile()
    return _compiled_pipeline


# ─── Public entry point ───────────────────────────────────────────────────────

def create_initial_state(
    user_id: str,
    fridge_image_b64: str | None,
    grocery_list: list[str],
    location_id: str,
    budget: float | None = None,
) -> GrocerAIState:
    """Build a fresh GrocerAIState for a new pipeline run."""
    return GrocerAIState(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        fridge_image_b64=fridge_image_b64,
        grocery_list=[item.strip() for item in grocery_list if item.strip()],
        location_id=location_id,
        budget=budget,
        detected_inventory=[],
        vision_raw_response=None,
        vision_error=None,
        grocery_needs=[],
        user_preferences={},
        gap_error=None,
        search_results={},
        price_summary={},
        search_error=None,
        cart_items=[],
        cart_total=0.0,
        checkout_result=None,
        cart_error=None,
        current_step="init",
        errors=[],
        warnings=[],
        recommendations=[],
        completed=False,
    )


def serialize_state(state: GrocerAIState) -> dict:
    """
    Serialize pipeline state for session persistence.
    Strips the large base64 image to keep the session payload small.
    """
    s = dict(state)
    s.pop("fridge_image_b64", None)  # don't persist large image data
    return s


def run_pipeline(
    user_id: str,
    fridge_image_b64: str | None,
    grocery_list: list[str],
    location_id: str,
    budget: float | None = None,
) -> GrocerAIState:
    """
    Main entry point. Runs the full 4-agent pipeline and returns final state.

    Args:
        user_id: Unique identifier for preference RAG lookup.
        fridge_image_b64: Base64-encoded fridge/pantry photo (optional).
        grocery_list: List of items the user wants to buy.
        location_id: Kroger store location ID for price lookup.
        budget: Optional spending limit in USD.

    Returns:
        Final GrocerAIState after all agents have run.
    """
    pipeline = build_pipeline()
    initial_state = create_initial_state(
        user_id, fridge_image_b64, grocery_list, location_id, budget
    )

    logger.info(
        "Starting GrocerAI pipeline | session=%s user=%s items=%d budget=%s",
        initial_state["session_id"],
        user_id,
        len(grocery_list),
        f"${budget:.2f}" if budget else "none",
    )

    start = time.time()
    success = False
    final_state = initial_state  # ensure always defined for finally block

    try:
        final_state = pipeline.invoke(initial_state)
        final_state["completed"] = True
        success = True

        logger.info(
            "Pipeline complete | session=%s errors=%d cart_items=%d total=$%.2f",
            final_state["session_id"],
            len(final_state.get("errors", [])),
            len(final_state.get("cart_items", [])),
            final_state.get("cart_total", 0.0),
        )
        return final_state

    except Exception as exc:
        logger.exception("Pipeline crashed | session=%s", initial_state["session_id"])
        initial_state["errors"].append(f"Pipeline crashed: {exc}")
        initial_state["completed"] = False
        raise

    finally:
        duration = time.time() - start
        record_pipeline_run(success=success, duration_seconds=duration)
        try:
            log_pipeline_run(
                final_state,
                duration_ms=int(duration * 1000),
            )
        except Exception:
            pass
