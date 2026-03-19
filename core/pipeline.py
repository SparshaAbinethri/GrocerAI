"""
core/pipeline.py
LangGraph graph definition — wires Vision → Gap → Search → Cart agents.
Each node is a pure function that receives and returns GrocerAIState.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from agents.vision_agent import run_vision_agent
from agents.gap_agent import run_gap_agent
from agents.search_agent import run_search_agent
from agents.cart_agent import run_cart_agent
from agents.recommendations_agent import run_recommendations_agent
from core.state import GrocerAIState
from core.metrics import record_pipeline_run, timer as metrics_timer
from core.db import log_pipeline_run

logger = logging.getLogger(__name__)


# ─── Node wrappers (add error isolation around each agent) ────────────────────

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
    try:
        return run_gap_agent(state)
    except Exception as exc:
        logger.exception("Gap Agent failed")
        state["gap_error"] = str(exc)
        state["errors"].append(f"Gap: {exc}")
        return state


def search_node(state: GrocerAIState) -> GrocerAIState:
    logger.info("[Pipeline] Running Search Agent")
    state["current_step"] = "search"
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
    try:
        return run_cart_agent(state)
    except Exception as exc:
        logger.exception("Cart Agent failed")
        state["cart_error"] = str(exc)
        state["errors"].append(f"Cart: {exc}")
        return state


# ─── Conditional edges ────────────────────────────────────────────────────────

def should_continue_after_vision(state: GrocerAIState) -> str:
    """Skip to END if vision failed and no fallback inventory."""
    if state.get("vision_error") and not state.get("detected_inventory"):
        logger.warning("Vision failed with no inventory — aborting pipeline")
        return "abort"
    return "continue"


def should_continue_after_gap(state: GrocerAIState) -> str:
    """Skip search/cart if nothing is needed."""
    needs = state.get("grocery_needs", [])
    if not any(n["needed"] for n in needs):
        logger.info("All items already stocked — skipping Search & Cart")
        return "fully_stocked"
    return "continue"


# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_pipeline() -> Any:
    """Compile and return the LangGraph pipeline."""
    graph = StateGraph(GrocerAIState)

    # Register nodes
    graph.add_node("vision", vision_node)
    graph.add_node("gap", gap_node)
    graph.add_node("search", search_node)
    graph.add_node("cart", cart_node)

    # Entry point
    graph.set_entry_point("vision")

    # Vision → Gap (conditional)
    graph.add_conditional_edges(
        "vision",
        should_continue_after_vision,
        {"continue": "gap", "abort": END},
    )

    # Gap → Search (conditional)
    graph.add_conditional_edges(
        "gap",
        should_continue_after_gap,
        {"continue": "search", "fully_stocked": END},
    )

    # Search → Cart → Recommendations → END (linear)
    graph.add_edge("search", "cart")
    graph.add_node("recommend", run_recommendations_agent)
    graph.add_edge("cart", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


# ─── Public entry point ───────────────────────────────────────────────────────

def create_initial_state(
    user_id: str,
    fridge_image_b64: str | None,
    grocery_list: list[str],
    location_id: str,
) -> GrocerAIState:
    """Build a fresh GrocerAIState for a new pipeline run."""
    return GrocerAIState(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        fridge_image_b64=fridge_image_b64,
        grocery_list=[item.strip() for item in grocery_list if item.strip()],
        location_id=location_id,
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


def run_pipeline(
    user_id: str,
    fridge_image_b64: str | None,
    grocery_list: list[str],
    location_id: str,
) -> GrocerAIState:
    """
    Main entry point. Runs the full 4-agent pipeline and returns final state.

    Args:
        user_id: Unique identifier for preference RAG lookup.
        fridge_image_b64: Base64-encoded fridge/pantry photo (optional).
        grocery_list: List of items the user wants to buy.
        location_id: Kroger store location ID for price lookup.

    Returns:
        Final GrocerAIState after all agents have run.
    """
    pipeline = build_pipeline()
    initial_state = create_initial_state(user_id, fridge_image_b64, grocery_list, location_id)

    logger.info(
        "Starting GrocerAI pipeline | session=%s user=%s items=%d",
        initial_state["session_id"],
        user_id,
        len(grocery_list),
    )

    import time
    start = time.time()
    success = False

    try:
        final_state: GrocerAIState = pipeline.invoke(initial_state)
        final_state["completed"] = True
        success = True

        logger.info(
            "Pipeline complete | session=%s errors=%d cart_items=%d",
            final_state["session_id"],
            len(final_state["errors"]),
            len(final_state.get("cart_items", [])),
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
                final_state if success else initial_state,
                duration_ms=int(duration * 1000),
            )
        except Exception:
            pass
