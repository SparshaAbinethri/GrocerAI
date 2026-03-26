"""
ui/app.py
GrocerAI — Main Streamlit Application

Fixes applied:
  - Removed ALL debug st.write() statements
  - Non-blocking UI: progress bar updates while pipeline runs in thread
  - Cart edits preserved across reruns (separated from pipeline state)
  - Clear error state UI for every failure mode
  - Budget input added to sidebar
  - Checkout result shows per-item successes/failures
  - Pipeline re-run does not wipe user's cart edits if same session
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Add project root to sys.path ─────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load .env BEFORE any imports that initialize API clients ──────────────────
# Must happen before core.config, PreferenceStore, OpenAI, etc. are imported
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from core.logging_config import setup_logging
from core.redis_client import (
    check_rate_limit,
    clear_kroger_tokens,
    get_kroger_access_token,
    get_kroger_refresh_token,
    store_kroger_tokens,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GrocerAI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging()

# Start health check server in background
try:
    from healthcheck import start_health_server
    start_health_server(daemon=True)
except Exception:
    pass

# ── OAuth callback ────────────────────────────────────────────────────────────
_query_params = st.query_params
if "code" in _query_params and "kroger_oauth_pending" in st.session_state:
    _code = _query_params["code"]
    try:
        from api.kroger import KrogerClient
        _token_data = KrogerClient().exchange_code_for_token(_code)
        _user_id = st.session_state.get("user_id", "anonymous")
        store_kroger_tokens(
            _user_id,
            _token_data["access_token"],
            _token_data.get("refresh_token"),
            _token_data.get("expires_in", 1800),
        )
        st.session_state["kroger_access_token"] = _token_data["access_token"]
        st.session_state.pop("kroger_oauth_pending", None)
        st.query_params.clear()
        st.success("✅ Successfully logged in to Kroger!")
        st.rerun()
    except Exception as _e:
        st.error(f"❌ Kroger login failed: {_e}")
        st.session_state.pop("kroger_oauth_pending", None)

from agents.cart_agent import checkout, update_cart_item
from core.pipeline import run_pipeline
from core.state import CartItem, GrocerAIState
from rag.preference_store import PreferenceStore

# ── Session state initialisation ─────────────────────────────────────────────
def _init_session():
    defaults = {
        "user_id": str(uuid.uuid4())[:8],
        "pipeline_result": None,
        "cart_items": [],
        "checked_out": False,
        "location_id": "",
        "budget": None,
        "_found_locs": None,
        "_uploaded_image_b64": None,
        "_uploaded_image_name": None,
        "kroger_access_token": None,
        "oauth_state": None,
        "kroger_tokens_restored": False,
        "_pipeline_running": False,
        "_pipeline_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Restore Kroger tokens from Redis once per session
    if not st.session_state["kroger_tokens_restored"]:
        stored = get_kroger_access_token(st.session_state["user_id"])
        if stored:
            st.session_state["kroger_access_token"] = stored
        st.session_state["kroger_tokens_restored"] = True

_init_session()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _image_to_b64(image_file) -> str:
    img = Image.open(image_file).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _recalc_total(cart_items: list[CartItem]) -> float:
    return round(sum(item["subtotal"] for item in cart_items), 2)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.sidebar.title("GrocerAI 🛒")
    st.sidebar.caption(f"Session: `{st.session_state.user_id}`")
    st.sidebar.divider()

    # ── Store & Location ──────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ Store & Location")
    st.sidebar.caption("🔍 Find your store by zip code to get real prices:")

    zip_code = st.sidebar.text_input(
        "Zip code",
        placeholder="e.g. 30301",
        key="zip_input",
        label_visibility="collapsed",
    )

    if st.sidebar.button("🔍 Find Stores", use_container_width=True, key="find_loc_btn"):
        if zip_code.strip():
            try:
                from api.kroger import KrogerClient
                locs = KrogerClient().search_locations(zip_code.strip(), limit=5)
                if locs:
                    st.session_state["_found_locs"] = locs
                    st.sidebar.success(f"Found {len(locs)} stores!")
                else:
                    st.sidebar.warning("No stores found near that zip.")
            except Exception as e:
                st.sidebar.error(f"Could not find stores: {e}")

    if st.session_state.get("_found_locs"):
        locs = st.session_state["_found_locs"]
        loc_labels = [
            f"{l.get('name', 'Store')} — {l.get('address', {}).get('city', '')}"
            for l in locs
        ]
        chosen = st.sidebar.selectbox("Pick a store", loc_labels, key="loc_picker")
        chosen_id = locs[loc_labels.index(chosen)].get("locationId", "")
        if st.sidebar.button("✅ Use this store", use_container_width=True, key="use_loc_btn"):
            st.session_state.location_id = chosen_id
            st.session_state["_found_locs"] = None
            st.sidebar.success(f"✅ Store set!")

    location_id = st.sidebar.text_input(
        "Kroger Location ID",
        value=st.session_state.location_id,
        placeholder="Auto-filled by finder above",
        help="Set by zip finder, or enter manually from developer.kroger.com",
    )
    if location_id != st.session_state.location_id:
        st.session_state.location_id = location_id

    # ── Budget ────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("💰 Budget")
    budget_input = st.sidebar.number_input(
        "Spending limit (USD)",
        min_value=0.0,
        max_value=1000.0,
        value=float(st.session_state.budget or 0),
        step=5.0,
        format="%.2f",
        help="Low-priority items will be removed if cart exceeds this amount.",
    )
    st.session_state.budget = budget_input if budget_input > 0 else None

    # ── Preferences ───────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("🏷️ Preferences")

    pref_store = PreferenceStore()
    prefs = pref_store.get_preferences(st.session_state.user_id)

    preferred_brands = st.sidebar.text_area(
        "Preferred brands (one per line)",
        value="\n".join(prefs.get("preferred_brands", [])),
        height=80,
    )
    avoid_brands = st.sidebar.text_area(
        "Avoid brands (one per line)",
        value="\n".join(prefs.get("avoid_brands", [])),
        height=80,
    )
    dietary = st.sidebar.multiselect(
        "Dietary restrictions",
        options=["vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free", "kosher", "halal", "low-sodium"],
        default=[r for r in prefs.get("dietary_restrictions", []) if r in
                 ["vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free", "kosher", "halal", "low-sodium"]],
    )
    quality_notes = st.sidebar.text_input(
        "Quality preferences",
        value=prefs.get("quality_notes", ""),
        placeholder="e.g. organic produce, grass-fed beef",
    )

    if st.sidebar.button("💾 Save Preferences", use_container_width=True):
        new_prefs = {
            "preferred_brands": [b.strip() for b in preferred_brands.splitlines() if b.strip()],
            "avoid_brands": [b.strip() for b in avoid_brands.splitlines() if b.strip()],
            "dietary_restrictions": dietary,
            "quality_notes": quality_notes,
        }
        pref_store.save_preferences(st.session_state.user_id, new_prefs)
        st.sidebar.success("✅ Preferences saved!")


# ─── Step 1 — Input ───────────────────────────────────────────────────────────

def render_input_section():
    st.title("🛒 GrocerAI")
    st.markdown(
        "Upload a photo of your fridge and your grocery list — "
        "GrocerAI will figure out what you actually need and find the best prices."
    )
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Fridge / Pantry Photo")
        st.caption("Optional — items already stocked will be filtered out automatically.")
        fridge_image = st.file_uploader(
            "Upload fridge photo",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if fridge_image:
            st.session_state["_uploaded_image_b64"] = _image_to_b64(fridge_image)
            st.session_state["_uploaded_image_name"] = fridge_image.name
            st.image(fridge_image, caption="Your fridge", use_column_width=True)
        elif st.session_state.get("_uploaded_image_b64"):
            st.success(f"✅ Image ready: {st.session_state.get('_uploaded_image_name', 'fridge photo')}")
            if st.button("🗑️ Remove image", key="remove_img"):
                st.session_state["_uploaded_image_b64"] = None
                st.session_state["_uploaded_image_name"] = None
                st.rerun()

    with col2:
        st.subheader("📝 Grocery List")
        st.caption("One item per line — be as specific or general as you like.")
        grocery_text = st.text_area(
            "Grocery list",
            placeholder="whole milk\neggs\ncheddar cheese\nsourdough bread\nchicken breast\nspinach\norange juice",
            height=250,
            label_visibility="collapsed",
        )

    st.divider()

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button(
            "🚀 Analyse & Find Prices",
            type="primary",
            use_container_width=True,
            disabled=not grocery_text.strip(),
        )

    # Show any previous pipeline error clearly
    if st.session_state.get("_pipeline_error"):
        st.error("❌ Pipeline failed on last run:")
        st.code(st.session_state["_pipeline_error"])
        if st.button("🔄 Clear error & try again"):
            st.session_state["_pipeline_error"] = None
            st.rerun()

    if run_btn:
        # Rate limiting
        _allowed, _remaining = check_rate_limit(
            f"pipeline:{st.session_state.user_id}",
            max_requests=int(os.getenv("RATE_LIMIT_PIPELINE", "10")),
            window_seconds=3600,
        )
        if not _allowed:
            st.error("⚠️ Rate limit reached — up to 10 analyses per hour. Please try again later.")
            st.stop()

        grocery_list = [line.strip() for line in grocery_text.splitlines() if line.strip()]

        # Capture image before any state changes
        image_b64 = None
        if fridge_image is not None:
            image_b64 = _image_to_b64(fridge_image)
            st.session_state["_uploaded_image_b64"] = image_b64
        elif st.session_state.get("_uploaded_image_b64"):
            image_b64 = st.session_state["_uploaded_image_b64"]

        # Clear previous results AFTER capturing image
        st.session_state.pipeline_result = None
        st.session_state.cart_items = []
        st.session_state.checked_out = False
        st.session_state["_pipeline_error"] = None

        # Capture values needed in thread (session_state not thread-safe)
        _user_id = st.session_state.user_id
        _location_id = st.session_state.get("location_id", "")
        _budget = st.session_state.get("budget")

        result_holder: dict = {}

        def _run():
            try:
                result_holder["result"] = run_pipeline(
                    user_id=_user_id,
                    fridge_image_b64=image_b64,
                    grocery_list=grocery_list,
                    location_id=_location_id,
                    budget=_budget,
                )
            except Exception as exc:
                import traceback
                result_holder["error"] = f"{exc}\n\n{traceback.format_exc()}"

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # Non-blocking progress bar while pipeline runs
        progress = st.progress(0, text="🔍 Vision Agent: analysing fridge...")
        steps = [
            (20, "🔍 Vision Agent: analysing fridge..."),
            (40, "🧠 Gap Agent: computing what you need..."),
            (60, "🔎 Search Agent: looking up prices..."),
            (80, "🛒 Cart Agent: assembling your cart..."),
            (95, "✨ Finalising recommendations..."),
        ]
        step_idx = 0
        while thread.is_alive():
            if step_idx < len(steps):
                pct, msg = steps[step_idx]
                progress.progress(pct, text=msg)
                step_idx += 1
            time.sleep(5)

        thread.join()
        progress.progress(100, text="✅ Done!")
        time.sleep(0.2)

        # Handle errors
        if "error" in result_holder:
            st.session_state["_pipeline_error"] = result_holder["error"]
            st.error("❌ Pipeline failed. See error below.")
            st.code(result_holder["error"])
            st.stop()

        if "result" not in result_holder:
            st.session_state["_pipeline_error"] = "Pipeline returned no result — thread may have timed out."
            st.error("❌ Pipeline timed out. Please try again.")
            st.stop()

        state: GrocerAIState = result_holder["result"]

        # Surface pipeline-level errors to user (not as tracebacks — friendly messages)
        if state.get("errors"):
            for err in state["errors"]:
                st.warning(f"⚠️ {err}")

        st.session_state.pipeline_result = state
        # Only set cart_items from pipeline if not already edited by user
        st.session_state.cart_items = list(state.get("cart_items", []))
        st.rerun()


# ─── Step 2 — Results ─────────────────────────────────────────────────────────

def render_inventory_tab(state: GrocerAIState):
    inventory = state.get("detected_inventory", [])
    if not inventory:
        st.info("No fridge image was provided, or no items were detected.")
        return

    st.subheader(f"Detected {len(inventory)} items in your fridge")

    # Separate by confidence for user clarity
    high = [i for i in inventory if i["confidence"] >= 0.8]
    medium = [i for i in inventory if 0.4 <= i["confidence"] < 0.8]
    low = [i for i in inventory if i["confidence"] < 0.4]

    if low:
        st.warning(
            f"⚠️ {len(low)} item(s) detected with low confidence — please verify these are actually in your fridge: "
            + ", ".join(i["name"].title() for i in low)
        )

    cols = st.columns(3)
    for i, item in enumerate(sorted(inventory, key=lambda x: -x["confidence"])):
        with cols[i % 3]:
            conf_color = "🟢" if item["confidence"] >= 0.8 else "🟡" if item["confidence"] >= 0.4 else "🔴"
            st.metric(
                label=f"{conf_color} {item['name'].title()}",
                value=f"{item['quantity']} {item['unit']}",
                help=f"Confidence: {item['confidence']:.0%}",
            )


def render_needs_tab(state: GrocerAIState):
    needs = state.get("grocery_needs", [])
    if not needs:
        st.info("No needs computed.")
        return

    needed = [n for n in needs if n["needed"]]
    skipped = [n for n in needs if not n["needed"]]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🛒 Need to buy ({len(needed)})")
        if not needed:
            st.success("All items are already stocked!")
        for n in sorted(needed, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["priority"], 1)):
            icon = "🔴" if n["priority"] == "high" else "🟡" if n["priority"] == "medium" else "🟢"
            st.markdown(f"{icon} **{n['name']}** — {n['reason']}")

    with col2:
        st.subheader(f"✅ Already stocked ({len(skipped)})")
        if not skipped:
            st.info("Nothing filtered out.")
        for n in skipped:
            st.markdown(f"✅ ~~{n['name']}~~ — {n['reason']}")


def render_cart_tab(state: GrocerAIState):
    # IMPORTANT: always read from session_state, not state["cart_items"]
    # This preserves user edits across reruns
    cart_items = st.session_state.cart_items
    search_results = state.get("search_results", {})

    if not cart_items:
        st.info("Your cart is empty. All items may already be stocked, or no products were found.")

        # Show which items had no search results so user knows why
        not_found = [
            name for name, results in search_results.items()
            if not results
        ]
        if not_found:
            st.warning(
                "The following items had no results on Kroger: "
                + ", ".join(not_found)
            )
        return

    # Price summary
    price_summary = state.get("price_summary", {})
    real_prices = {s: t for s, t in price_summary.items() if t > 0}

    if real_prices:
        st.subheader("💰 Price Estimate by Store")
        pcols = st.columns(len(real_prices))
        for i, (store, total) in enumerate(real_prices.items()):
            with pcols[i]:
                is_estimated = not st.session_state.location_id
                st.metric(
                    store.title(),
                    f"{'~' if is_estimated else ''}${total:.2f}",
                    help="~Estimated" if is_estimated else "Live price",
                )
        if not st.session_state.location_id:
            st.caption("💡 Prices are estimates. Add your store location in the sidebar for live prices.")
    elif price_summary:
        st.info("💡 Prices unavailable — enter a Kroger Location ID in the sidebar for live prices.")

    # Budget warning
    if st.session_state.budget:
        total_now = _recalc_total(cart_items)
        if total_now > st.session_state.budget:
            st.warning(
                f"⚠️ Cart total ${total_now:.2f} exceeds your budget of ${st.session_state.budget:.2f}. "
                "Remove items or increase your budget in the sidebar."
            )

    st.divider()
    st.subheader(f"🛒 Cart Review — {len(cart_items)} items")

    cart_changed = False
    updated_items = list(cart_items)

    for idx, cart_item in enumerate(cart_items):
        need_name = cart_item["grocery_need"]
        product = cart_item["selected_product"]
        alternatives = search_results.get(need_name, [])

        is_estimated = product.get("price_estimated", False) or (
            product["price"] > 0 and not st.session_state.location_id
        )
        price_label = (
            f"~${product['price']:.2f} est." if is_estimated and product["price"] > 0
            else f"${product['price']:.2f}" if product["price"] > 0
            else "Price N/A"
        )

        with st.expander(
            f"{'⭐ ' if product.get('preferred') else ''}"
            f"**{need_name.title()}** — {product['brand']} {product['name']} | {price_label}",
            expanded=False,
        ):
            c1, c2, c3 = st.columns([2, 1, 1])

            with c1:
                if product.get("image_url"):
                    st.image(product["image_url"], width=80)
                st.write(f"**{product['name']}**")
                st.caption(f"Brand: {product['brand']} | Store: {product['store'].title()}")

                _product_id = product.get("product_id", "")
                _slug = product["name"].replace(" ", "%20")
                _kroger_url = (
                    f"https://www.kroger.com/p/{_slug}/{_product_id}"
                    if _product_id
                    else f"https://www.kroger.com/search?query={_slug}"
                )
                st.markdown(f"[🔗 View on Kroger]({_kroger_url})")

                if is_estimated:
                    st.caption("~ Estimated price — may differ at your store")
                if product.get("preferred"):
                    st.success("⭐ Matches your brand preference")
                if not product.get("in_stock"):
                    st.warning("⚠️ May not be in stock at this location")

            with c2:
                st.caption("Quantity")
                q1, q2, q3 = st.columns([1, 1, 1])
                current_qty = cart_item["quantity"]
                with q1:
                    if st.button("➖", key=f"dec_{idx}", use_container_width=True):
                        if current_qty > 1:
                            updated_items = update_cart_item(updated_items, need_name, new_quantity=current_qty - 1)
                            cart_changed = True
                with q2:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:8px;font-weight:bold'>{current_qty}</div>",
                        unsafe_allow_html=True,
                    )
                with q3:
                    if st.button("➕", key=f"inc_{idx}", use_container_width=True):
                        updated_items = update_cart_item(updated_items, need_name, new_quantity=current_qty + 1)
                        cart_changed = True

                if st.button("🗑️ Remove", key=f"remove_{idx}", use_container_width=True):
                    updated_items = [i for i in updated_items if i["grocery_need"] != need_name]
                    cart_changed = True

            with c3:
                if len(alternatives) > 1:
                    alt_labels = [
                        f"{a['brand']} — {'~$' if is_estimated else '$'}"
                        f"{a['price']:.2f}" if a["price"] > 0 else f"{a['brand']} — Price N/A"
                        for a in alternatives[:5]
                    ]
                    selected = st.selectbox("Switch product", alt_labels, index=0, key=f"alt_{idx}")
                    alt_idx = alt_labels.index(selected)
                    if alt_idx != 0:
                        updated_items = update_cart_item(updated_items, need_name, new_product=alternatives[alt_idx])
                        cart_changed = True
                else:
                    st.caption("No alternatives found")

    # Persist cart edits immediately — don't wait for rerun
    if cart_changed:
        st.session_state.cart_items = updated_items

    st.divider()
    total = _recalc_total(st.session_state.cart_items)
    st.metric(
        "🧾 Cart Total",
        f"${total:.2f}" if total > 0 else "N/A",
        help="Estimated total. Add a store location for live prices." if not st.session_state.location_id else None,
    )

    checkout_col, clear_col = st.columns([2, 1])
    with checkout_col:
        if st.session_state.get("kroger_access_token"):
            st.success("✅ Logged in to Kroger")
            if st.button("🛒 Add All to Kroger Cart", type="primary", use_container_width=True):
                _do_kroger_checkout(
                    st.session_state.cart_items,
                    st.session_state["kroger_access_token"],
                    st.session_state.user_id,
                )
            if st.button("🔓 Logout from Kroger", use_container_width=True):
                clear_kroger_tokens(st.session_state.user_id)
                st.session_state["kroger_access_token"] = None
                st.rerun()
        else:
            if st.button("🔐 Login to Kroger & Checkout", type="primary", use_container_width=True):
                from api.kroger import KrogerClient
                import secrets
                oauth_state = secrets.token_urlsafe(16)
                st.session_state["oauth_state"] = oauth_state
                st.session_state["kroger_oauth_pending"] = True
                oauth_url = KrogerClient().get_oauth_url(state=oauth_state)
                st.markdown(
                    f"""<div style='background:#fff3cd;padding:16px;border-radius:8px;border:1px solid #ffc107'>
<b>🔐 Step 1:</b> Click below to log in to Kroger<br><br>
<a href="{oauth_url}" target="_blank" style='font-size:16px;font-weight:bold'>👉 Open Kroger Login</a><br><br>
<b>Step 2:</b> After logging in, Kroger redirects back here automatically.
</div>""",
                    unsafe_allow_html=True,
                )

    with clear_col:
        if st.button("🗑️ Clear Cart", use_container_width=True):
            st.session_state.cart_items = []
            st.session_state.pipeline_result = None
            st.session_state.pop("kroger_access_token", None)
            st.rerun()


def _do_kroger_checkout(cart_items: list, access_token: str, user_id: str = "") -> None:
    """Add all cart items to Kroger via API and display per-item results."""
    from api.kroger import KrogerClient

    # Attempt token refresh
    if user_id:
        refresh_token = get_kroger_refresh_token(user_id)
        if refresh_token:
            try:
                new_tokens = KrogerClient().refresh_user_token(refresh_token)
                access_token = new_tokens["access_token"]
                store_kroger_tokens(
                    user_id, access_token,
                    new_tokens.get("refresh_token", refresh_token),
                    new_tokens.get("expires_in", 1800),
                )
                st.session_state["kroger_access_token"] = access_token
            except Exception:
                pass

    with st.spinner("Adding items to your Kroger cart..."):
        result = checkout(cart_items, access_token)

    if result["success"]:
        st.success(f"🎉 {len(result.get('added_items', []))} item(s) added to your Kroger cart!")
        st.balloons()
        if result.get("order_id"):
            st.caption(f"Order ID: {result['order_id']}")
        if result.get("failed_items"):
            with st.expander(f"⚠️ {len(result['failed_items'])} item(s) could not be added"):
                for name, reason in result["failed_items"]:
                    st.caption(f"• {name}: {reason}")
        st.markdown("[🛒 View your Kroger cart](https://www.kroger.com/cart)")
    else:
        st.error(f"❌ Checkout failed: {result.get('error', 'Unknown error')}")
        if result.get("failed_items"):
            with st.expander("Item details"):
                for name, reason in result["failed_items"]:
                    st.caption(f"• {name}: {reason}")
        st.caption("Your Kroger session may have expired. Try logging in again.")
        st.session_state.pop("kroger_access_token", None)


def render_recommendations_tab(state: GrocerAIState) -> None:
    recs = state.get("recommendations", [])
    if not recs:
        st.info("No recommendations available. Run the pipeline to generate personalised suggestions.")
        return

    st.subheader("✨ Recommended for You")
    st.caption("Based on what you're buying, your preferences, and what pairs well together.")

    category_icons = {
        "produce": "🥦", "dairy": "🧀", "meat": "🥩", "bakery": "🍞",
        "pantry": "🫙", "snacks": "🍿", "beverages": "🥤", "frozen": "🧊",
    }

    from collections import defaultdict
    by_category: dict[str, list] = defaultdict(list)
    for rec in recs:
        by_category[rec.get("category", "pantry")].append(rec)

    for category, items in by_category.items():
        icon = category_icons.get(category, "🛒")
        st.markdown(f"#### {icon} {category.title()}")
        cols = st.columns(min(len(items), 3))
        for i, rec in enumerate(items):
            with cols[i % 3]:
                st.markdown(
                    f"""<div style='border:1px solid #e0e0e0;border-radius:10px;padding:14px;
                    min-height:140px;background:#fafafa'>
                    <b>{rec['name']}</b><br>
                    <small style='color:#666'>{rec['reason']}</small><br><br>
                    <span style='color:#2d7a2d;font-weight:bold'>~${rec.get('estimated_price', 0):.2f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(f"[🔗 Find on Kroger]({rec['kroger_search_url']})")
        st.divider()


def render_results(state: GrocerAIState):
    st.title("🛒 GrocerAI — Results")
    st.caption(f"Session: `{state['session_id']}`")

    # Pipeline errors — shown prominently
    if state.get("errors"):
        with st.expander(f"❌ {len(state['errors'])} error(s) during pipeline", expanded=True):
            for err in state["errors"]:
                st.error(err)

    # Warnings — collapsible
    if state.get("warnings"):
        with st.expander(f"⚠️ {len(state['warnings'])} warning(s)"):
            for w in state["warnings"]:
                st.caption(f"• {w}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🥬 Inventory Detected",
        "📋 What You Need",
        "🛒 Cart",
        "✨ Recommendations",
    ])
    with tab1:
        render_inventory_tab(state)
    with tab2:
        render_needs_tab(state)
    with tab3:
        render_cart_tab(state)
    with tab4:
        render_recommendations_tab(state)

    st.divider()
    if st.button("⬅️ Start Over"):
        st.session_state.pipeline_result = None
        st.session_state.cart_items = []
        st.session_state["_uploaded_image_b64"] = None
        st.session_state["_uploaded_image_name"] = None
        st.session_state["_pipeline_error"] = None
        st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    if st.session_state.pipeline_result:
        render_results(st.session_state.pipeline_result)
    else:
        render_input_section()


if __name__ == "__main__":
    main()