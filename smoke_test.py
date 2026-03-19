"""
smoke_test.py
Run this BEFORE launching the full app to verify your credentials work.

Usage:
    python smoke_test.py

Checks:
  1. Kroger OAuth token fetch
  2. Kroger product search (no location needed)
  3. OpenAI API connectivity
  4. FAISS index creation
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []

print("\n🛒 GrocerAI Smoke Test")
print("=" * 40)


# ── 1. Env vars ───────────────────────────────────────────────────────────────
print("\n[1/4] Checking environment variables...")

required = ["OPENAI_API_KEY", "KROGER_CLIENT_ID", "KROGER_CLIENT_SECRET"]
for var in required:
    val = os.getenv(var)
    if val:
        print(f"  {PASS} {var} = {val[:8]}...")
    else:
        print(f"  {FAIL} {var} is not set")
        errors.append(f"Missing env var: {var}")

location_id = os.getenv("KROGER_LOCATION_ID", "")
if location_id:
    print(f"  {PASS} KROGER_LOCATION_ID = {location_id}")
else:
    print(f"  {WARN} KROGER_LOCATION_ID not set — prices won't show (sandbox mode)")


# ── 2. Kroger auth + product search ───────────────────────────────────────────
print("\n[2/4] Testing Kroger API...")
try:
    from api.kroger import KrogerClient
    client = KrogerClient()
    token = client._get_app_token()
    print(f"  {PASS} OAuth token fetched: {token[:16]}...")

    results = client.search_products("milk", location_id=location_id, limit=3)
    print(f"  {PASS} Product search returned {len(results)} results")
    if results:
        first = results[0]
        name = first.get("description", "unknown")
        brand = first.get("brand", "unknown")
        items = first.get("items", [{}])
        price = (items[0].get("price") or {}).get("regular") if items else None
        price_str = f"${price:.2f}" if price else "no price (sandbox)"
        print(f"       First result: {brand} — {name} {price_str}")
    else:
        print(f"  {WARN} No results returned — check your client credentials")

except Exception as e:
    print(f"  {FAIL} Kroger API failed: {e}")
    errors.append(f"Kroger API: {e}")


# ── 3. OpenAI connectivity ────────────────────────────────────────────────────
print("\n[3/4] Testing OpenAI API...")
try:
    from openai import OpenAI
    oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = oai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Reply with just the word: OK"}],
        max_tokens=5,
    )
    reply = resp.choices[0].message.content.strip()
    print(f"  {PASS} GPT-4o responded: '{reply}'")
except Exception as e:
    print(f"  {FAIL} OpenAI API failed: {e}")
    errors.append(f"OpenAI: {e}")


# ── 4. FAISS index ────────────────────────────────────────────────────────────
print("\n[4/4] Testing FAISS preference store...")
try:
    from rag.preference_store import PreferenceStore
    store = PreferenceStore()
    store.save_preferences("smoke_test_user", {
        "preferred_brands": ["tillamook"],
        "avoid_brands": [],
        "dietary_restrictions": ["gluten-free"],
        "quality_notes": "organic preferred",
    })
    prefs = store.get_preferences("smoke_test_user")
    assert prefs["preferred_brands"] == ["tillamook"]
    print(f"  {PASS} FAISS store created and preferences saved/retrieved")
except Exception as e:
    print(f"  {FAIL} FAISS store failed: {e}")
    errors.append(f"FAISS: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
if errors:
    print(f"❌ {len(errors)} issue(s) found:\n")
    for err in errors:
        print(f"   • {err}")
    print("\nFix the above before running the full app.")
    sys.exit(1)
else:
    print("✅ All checks passed — ready to launch!")
    print("\nRun the app with:")
    print("  streamlit run ui/app.py")
    print("  — or —")
    print("  docker-compose up --build")
