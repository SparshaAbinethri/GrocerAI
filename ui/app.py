"""
ui/app.py
GrocerAI — Main Streamlit Application
Full consumer-grade shopping UI with:
  - Fridge photo upload
  - Grocery list input
  - Real-time pipeline progress
  - Editable cart review with brand overrides
  - Price comparison by store
  - Persistent preference management
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Add project root to sys.path so all packages (agents, core, rag, api) ──────
# are importable regardless of which directory Streamlit is launched from.
# Works on Windows, Mac, Linux, and inside Streamlit's exec() context.
_HERE = Path(__file__).resolve().parent          # .../GrocerAI/ui
_ROOT = _HERE.parent                              # .../GrocerAI
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import base64
import uuid
from io import BytesIO
from typing import Any

import streamlit as st
from PIL import Image
from core.logging_config import setup_logging
from core.redis_client import (
    store_kroger_tokens, get_kroger_access_token,
    get_kroger_refresh_token, clear_kroger_tokens, check_rate_limit,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GrocerAI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Startup ──────────────────────────────────────────────────────────────────
setup_logging()

# Start health check server in background (non-blocking)
try:
    from healthcheck import start_health_server
    start_health_server(daemon=True)
except Exception:
    pass

# ── OAuth callback: auto-read ?code= from redirect URL ───────────────────────
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
        st.session_state["kroger_refresh_token"] = _token_data.get("refresh_token")
        st.session_state.pop("kroger_oauth_pending", None)
        # Clear code from URL
        st.query_params.clear()
        st.success("✅ Successfully logged in to Kroger!")
        st.rerun()
    except Exception as _e:
        st.error(f"❌ Kroger login failed: {_e}")
        st.session_state.pop("kroger_oauth_pending", None)

from agents.cart_agent import update_cart_item
from core.pipeline import run_pipeline
from core.state import CartItem, GrocerAIState
from rag.preference_store import PreferenceStore

# ── Session state initialisation ─────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

# Restore Kroger tokens from Redis if session is fresh
if "kroger_tokens_restored" not in st.session_state:
    _stored_token = get_kroger_access_token(st.session_state.get("user_id", ""))
    if _stored_token:
        st.session_state["kroger_access_token"] = _stored_token
    st.session_state["kroger_tokens_restored"] = True

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "cart_items" not in st.session_state:
    st.session_state.cart_items = []

if "checked_out" not in st.session_state:
    st.session_state.checked_out = False

if "location_id" not in st.session_state:
    st.session_state.location_id = ""

if "_found_locs" not in st.session_state:
    st.session_state["_found_locs"] = None

if "kroger_access_token" not in st.session_state:
    st.session_state["kroger_access_token"] = None

if "oauth_state" not in st.session_state:
    st.session_state["oauth_state"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_b64(image_file) -> str:
    img = Image.open(image_file).convert("RGB")
    # Resize to max 1024px on longest side to keep tokens reasonable
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _recalc_total(cart_items: list[CartItem]) -> float:
    return round(sum(item["subtotal"] for item in cart_items), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Preferences
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.sidebar.title("GrocerAI 🛒")
    st.sidebar.caption(f"Session: `{st.session_state.user_id}`")
    st.sidebar.divider()

    st.sidebar.subheader("⚙️ Store & Location")

    # ── Zip code store finder ─────────────────────────────────────────────────
    st.sidebar.caption("🔍 Find your store by zip code to get real prices:")
    zip_code = st.sidebar.text_input(
        "Zip code",
        placeholder="e.g. 30301",
        key="zip_input",
        label_visibility="collapsed",
    )
    find_btn = st.sidebar.button("🔍 Find Stores", use_container_width=True, key="find_loc_btn")

    if find_btn and zip_code.strip():
        try:
            from api.kroger import KrogerClient
            client = KrogerClient()
            locs = client.search_locations(zip_code.strip(), limit=5)
            if locs:
                st.session_state["_found_locs"] = locs
                st.sidebar.success(f"Found {len(locs)} stores!")
            else:
                st.sidebar.warning("No stores found near that zip.")
        except Exception as e:
            st.sidebar.error(f"Error finding stores: {e}")

    if st.session_state.get("_found_locs"):
        locs = st.session_state["_found_locs"]
        loc_labels = [
            f"{l.get('name','Store')} — {l.get('address',{}).get('city','')}"
            for l in locs
        ]
        chosen = st.sidebar.selectbox("Pick a store", loc_labels, key="loc_picker")
        chosen_id = locs[loc_labels.index(chosen)].get("locationId", "")
        if st.sidebar.button("✅ Use this store", use_container_width=True, key="use_loc_btn"):
            st.session_state.location_id = chosen_id
            st.session_state["_found_locs"] = None
            st.rerun()

    location_id = st.sidebar.text_input(
        "Kroger Location ID",
        value=st.session_state.location_id,
        placeholder="Auto-filled by finder above",
        help="Set by zip finder, or enter manually from developer.kroger.com",
    )
    if location_id != st.session_state.location_id:
        st.session_state.location_id = location_id

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
        default=prefs.get("dietary_restrictions", []),
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
        st.sidebar.success("Preferences saved!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Input
# ─────────────────────────────────────────────────────────────────────────────

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
        st.caption("Optional — if provided, items already stocked will be filtered out automatically.")
        fridge_image = st.file_uploader(
            "Upload fridge photo",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if fridge_image:
            st.image(fridge_image, caption="Your fridge", use_column_width=True)

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

    if run_btn:
        # ── Rate limiting ─────────────────────────────────────────────────────
        _user_id_rl = st.session_state.user_id
        _allowed, _remaining = check_rate_limit(
            f"pipeline:{_user_id_rl}",
            max_requests=int(os.getenv("RATE_LIMIT_PIPELINE", "10")),
            window_seconds=3600,
        )
        if not _allowed:
            st.error("⚠️ Rate limit reached — you can run up to 10 analyses per hour. Please try again later.")
            st.stop()

        grocery_list = [line.strip() for line in grocery_text.splitlines() if line.strip()]

        image_b64 = None
        if fridge_image:
            with st.spinner("Processing image..."):
                image_b64 = _image_to_b64(fridge_image)

        st.session_state.pipeline_result = None
        st.session_state.cart_items = []
        st.session_state.checked_out = False

        with st.spinner("Running GrocerAI pipeline... this may take 20-30 seconds"):
            progress = st.progress(0, text="Vision Agent: analysing fridge...")

            # We can't truly stream LangGraph, so simulate step progress
            import threading
            import time

            # Capture session state values BEFORE spawning thread
            # st.session_state is not accessible inside threads
            _user_id = st.session_state.user_id
            _location_id = st.session_state.get("location_id", "")

            result_holder = {}

            def _run():
                try:
                    result_holder["result"] = run_pipeline(
                        user_id=_user_id,
                        fridge_image_b64=image_b64,
                        grocery_list=grocery_list,
                        location_id=_location_id,
                    )
                except Exception as exc:
                    import traceback
                    result_holder["error"] = f"{exc}\n\n{traceback.format_exc()}"

            thread = threading.Thread(target=_run)
            thread.start()

            steps = [
                (25, "Gap Agent: computing what you need..."),
                (50, "Search Agent: looking up prices..."),
                (75, "Cart Agent: assembling your cart..."),
                (95, "Finalising..."),
            ]
            i = 0
            while thread.is_alive():
                if i < len(steps):
                    pct, msg = steps[i]
                    progress.progress(pct, text=msg)
                    i += 1
                time.sleep(4)

            thread.join()
            progress.progress(100, text="Done!")
            time.sleep(0.3)

        # Surface thread errors clearly instead of cryptic KeyError
        if "error" in result_holder:
            st.error("❌ Pipeline failed:")
            st.code(result_holder["error"])
            st.stop()

        state: GrocerAIState = result_holder["result"]
        st.session_state.pipeline_result = state
        st.session_state.cart_items = list(state.get("cart_items", []))

        if state.get("errors"):
            for err in state["errors"]:
                st.error(f"⚠️ {err}")

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Results
# ─────────────────────────────────────────────────────────────────────────────

def render_inventory_tab(state: GrocerAIState):
    inventory = state.get("detected_inventory", [])
    if not inventory:
        st.info("No fridge image was provided, or no items were detected.")
        return

    st.subheader(f"Detected {len(inventory)} items in your fridge")

    cols = st.columns(3)
    for i, item in enumerate(inventory):
        with cols[i % 3]:
            conf_color = "🟢" if item["confidence"] >= 0.8 else "🟡" if item["confidence"] >= 0.5 else "🔴"
            st.metric(
                label=item["name"].title(),
                value=f"{item['quantity']} {item['unit']}",
                help=f"Confidence: {item['confidence']:.0%} {conf_color}",
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
        for n in sorted(needed, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]]):
            icon = "🔴" if n["priority"] == "high" else "🟡" if n["priority"] == "medium" else "🟢"
            st.markdown(f"{icon} **{n['name']}** — {n['reason']}")

    with col2:
        st.subheader(f"✅ Already stocked ({len(skipped)})")
        for n in skipped:
            st.markdown(f"✅ ~~{n['name']}~~ — {n['reason']}")


def render_cart_tab(state: GrocerAIState):
    cart_items = st.session_state.cart_items
    search_results = state.get("search_results", {})

    if not cart_items:
        st.info("Your cart is empty. All items may already be stocked, or no products were found.")
        return

    # Price summary
    price_summary = state.get("price_summary", {})
    real_prices = {s: t for s, t in price_summary.items() if t > 0}
    if real_prices:
        st.subheader("💰 Price Estimate by Store")
        pcols = st.columns(len(real_prices))
        for i, (store, total) in enumerate(real_prices.items()):
            with pcols[i]:
                st.metric(store.title(), f"${total:.2f}", help="Estimated total for needed items")
    elif price_summary:
        st.info("💡 **Prices unavailable** — Kroger sandbox credentials don't include pricing. Use the zip finder in the sidebar to select a real store location and get live prices.")

    st.divider()
    st.subheader(f"🛒 Cart Review — {len(cart_items)} items")

    cart_changed = False
    updated_items = list(cart_items)

    for idx, cart_item in enumerate(cart_items):
        need_name = cart_item["grocery_need"]
        product = cart_item["selected_product"]
        alternatives = search_results.get(need_name, [])

        # ── Price label: show ~estimated if no real price ────────────────────
        _is_estimated = product.get("price_estimated", False) or (product["price"] > 0 and not st.session_state.location_id)
        _price_label = (
            f"~${product['price']:.2f} est." if _is_estimated and product["price"] > 0
            else f"${product['price']:.2f}" if product["price"] > 0
            else "Price N/A"
        )

        with st.expander(
            f"{'⭐' if product['preferred'] else '  '} **{need_name.title()}** — "
            f"{product['brand']} {product['name']} | {_price_label}",
            expanded=False,
        ):
            c1, c2, c3 = st.columns([2, 1, 1])

            with c1:
                if product.get("image_url"):
                    st.image(product["image_url"], width=80)
                st.write(f"**{product['name']}**")
                st.caption(f"Brand: {product['brand']} | Store: {product['store'].title()}")

                # ── Kroger product link ───────────────────────────────────────
                _product_id = product.get("product_id", "")
                _product_name_slug = product["name"].replace(" ", "%20")
                if _product_id:
                    _kroger_url = f"https://www.kroger.com/p/{_product_name_slug}/{_product_id}"
                else:
                    _kroger_url = f"https://www.kroger.com/search?query={_product_name_slug}"
                st.markdown(f"[🔗 View on Kroger]({_kroger_url})", unsafe_allow_html=False)

                if _is_estimated and product["price"] > 0:
                    st.caption("~️ Estimated price — may differ at your store")
                elif not st.session_state.location_id:
                    st.caption("💡 Add a Location ID in the sidebar for real prices")
                if product.get("preferred"):
                    st.success("⭐ Matches your brand preference")
                if not product.get("in_stock"):
                    st.warning("⚠️ May not be in stock")

            with c2:
                st.caption("Quantity")
                qty_col1, qty_col2, qty_col3 = st.columns([1, 1, 1])
                current_qty = cart_item["quantity"]
                with qty_col1:
                    if st.button("➖", key=f"dec_{idx}", use_container_width=True):
                        if current_qty > 1:
                            updated_items = update_cart_item(updated_items, need_name, new_quantity=current_qty - 1)
                            cart_changed = True
                with qty_col2:
                    st.markdown(f"<div style='text-align:center;padding-top:8px;font-weight:bold'>{current_qty}</div>", unsafe_allow_html=True)
                with qty_col3:
                    if st.button("➕", key=f"inc_{idx}", use_container_width=True):
                        updated_items = update_cart_item(updated_items, need_name, new_quantity=current_qty + 1)
                        cart_changed = True
                # Remove item button
                if st.button("🗑️ Remove", key=f"remove_{idx}", use_container_width=True):
                    updated_items = [i for i in updated_items if i["grocery_need"] != need_name]
                    cart_changed = True

            with c3:
                if alternatives:
                    alt_labels = [
                        f"{a['brand']} {'~${:.2f}'.format(a['price']) if _is_estimated and a['price'] > 0 else '${:.2f}'.format(a['price']) if a['price'] > 0 else '(price N/A)'}"
                        for a in alternatives[:5]
                    ]
                    selected_alt_label = st.selectbox(
                        "Switch product",
                        options=alt_labels,
                        index=0,
                        key=f"alt_{idx}",
                    )
                    alt_idx = alt_labels.index(selected_alt_label)
                    if alt_idx != 0:
                        updated_items = update_cart_item(
                            updated_items,
                            need_name,
                            new_product=alternatives[alt_idx],
                        )
                        cart_changed = True

    if cart_changed:
        st.session_state.cart_items = updated_items

    st.divider()
    total = _recalc_total(st.session_state.cart_items)
    if total == 0.0:
        st.metric("🧾 Cart Total", "N/A")
        st.caption("💡 Prices unavailable in sandbox mode. Enter a Kroger Location ID in the sidebar to get real prices.")
    else:
        st.metric("🧾 Cart Total", f"${total:.2f}")

    checkout_col, clear_col = st.columns([2, 1])
    with checkout_col:
        if st.session_state.get("kroger_access_token"):
            # ── Already authenticated ─────────────────────────────────────────
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
            # ── OAuth flow ────────────────────────────────────────────────────
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
<a href="{oauth_url}" target="_blank" style='font-size:16px;font-weight:bold'>
👉 Open Kroger Login
</a><br><br>
<b>Step 2:</b> After logging in, Kroger will redirect you back to this page automatically.<br>
<small>The page will refresh and your cart will be submitted.</small>
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
    """Add all cart items to Kroger via API and show result."""
    from api.kroger import KrogerClient
    from agents.cart_agent import checkout

    # Try token refresh if needed
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
                pass  # Use existing token

    with st.spinner("Adding items to your Kroger cart..."):
        result = checkout(cart_items, access_token)

    if result["success"]:
        st.success("🎉 Items added to your Kroger cart!")
        st.balloons()
        if result.get("order_id"):
            st.caption(f"Order ID: {result['order_id']}")
        st.markdown(
            "[🛒 View your Kroger cart](https://www.kroger.com/cart)",
            unsafe_allow_html=False,
        )
    else:
        st.error(f"❌ Checkout failed: {result.get('error', 'Unknown error')}")
        st.caption("Your Kroger session may have expired. Try logging in again.")
        st.session_state.pop("kroger_access_token", None)


def render_recommendations_tab(state: GrocerAIState) -> None:
    """Render personalised product recommendations."""
    recs = state.get("recommendations", [])

    if not recs:
        st.info("No recommendations available. Run the pipeline to generate personalised suggestions.")
        return

    st.subheader("✨ Recommended for You")
    st.caption("Based on what you're buying, your preferences, and what pairs well together.")

    # Category icons
    category_icons = {
        "produce": "🥦", "dairy": "🧀", "meat": "🥩", "bakery": "🍞",
        "pantry": "🫙", "snacks": "🍿", "beverages": "🥤", "frozen": "🧊",
    }

    # Group by category
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
                with st.container():
                    st.markdown(
                        f"""
                        <div style='border:1px solid #e0e0e0;border-radius:10px;padding:14px;height:160px;background:#fafafa'>
                        <b>{rec['name']}</b><br>
                        <small style='color:#666'>{rec['reason']}</small><br><br>
                        <span style='color:#2d7a2d;font-weight:bold'>~${rec.get('estimated_price', 0):.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"[🔗 Find on Kroger]({rec['kroger_search_url']})",
                        unsafe_allow_html=False,
                    )
        st.divider()

    # Add all recommendations to cart shortcut
    if st.button("➕ Add All Recommendations to List", use_container_width=True):
        st.info("💡 Tip: Re-run the pipeline with these items added to your grocery list to include them in your cart.")


def render_results(state: GrocerAIState):
    st.title("🛒 GrocerAI — Results")
    st.caption(f"Session: `{state['session_id']}`")

    if state.get("warnings"):
        with st.expander(f"⚠️ {len(state['warnings'])} warnings"):
            for w in state["warnings"]:
                st.caption(w)

    tab1, tab2, tab3, tab4 = st.tabs(["🥬 Inventory Detected", "📋 What You Need", "🛒 Cart", "✨ Recommendations"])

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
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    if st.session_state.pipeline_result:
        render_results(st.session_state.pipeline_result)
    else:
        render_input_section()


if __name__ == "__main__":
    main()
