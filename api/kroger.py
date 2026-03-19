"""
api/kroger.py
Kroger API client — handles OAuth2 client credentials flow (for product search)
and user-delegated auth (for cart operations). Tokens are cached in-process.
Docs: https://developer.kroger.com/api-products/api/product-api
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from core.config import settings

logger = logging.getLogger(__name__)


class _TokenCache:
    """Simple in-process token cache with expiry."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0

    def get(self) -> str | None:
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        return None

    def set(self, token: str, expires_in: int) -> None:
        self._token = token
        self._expires_at = time.time() + expires_in


_app_token_cache = _TokenCache()


class KrogerClient:
    """
    Thin wrapper around the Kroger API.

    Product search uses client-credentials app token (no user login needed).
    Cart operations require a user OAuth token passed explicitly.
    """

    BASE_URL = settings.kroger_base_url
    AUTH_URL = "https://api.kroger.com/v1/connect/oauth2/token"

    def _get_app_token(self) -> str:
        """Fetch or return cached client-credentials token."""
        cached = _app_token_cache.get()
        if cached:
            return cached

        resp = requests.post(
            self.AUTH_URL,
            data={"grant_type": "client_credentials", "scope": "product.compact"},
            auth=HTTPBasicAuth(settings.kroger_client_id, settings.kroger_client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data["access_token"]
        expires_in = int(data.get("expires_in", 1800))
        _app_token_cache.set(token, expires_in)
        logger.debug("Fetched new Kroger app token (expires_in=%d)", expires_in)
        return token

    def _headers(self, token: str | None = None) -> dict[str, str]:
        tok = token or self._get_app_token()
        return {
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
        }

    # ── Product Search ────────────────────────────────────────────────────────

    def search_products(
        self,
        query: str,
        location_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for products by keyword. location_id is optional —
        omitting it works with sandbox credentials but won't return
        pricing (prices require an approved production app with location access).
        Returns raw Kroger product objects (list).
        """
        params: dict[str, Any] = {
            "filter.term": query,
            "filter.limit": limit,
        }
        if location_id:
            params["filter.locationId"] = location_id

        resp = requests.get(
            f"{self.BASE_URL}/products",
            headers=self._headers(),
            params=params,
            timeout=15,
        )

        if resp.status_code == 401:
            # Token might be stale — bust cache and retry once
            _app_token_cache._token = None
            resp = requests.get(
                f"{self.BASE_URL}/products",
                headers=self._headers(),
                params=params,
                timeout=15,
            )

        resp.raise_for_status()
        return resp.json().get("data", [])

    def get_product(self, product_id: str, location_id: str) -> dict[str, Any] | None:
        """Fetch a single product by ID."""
        params = {}
        if location_id:
            params["filter.locationId"] = location_id

        resp = requests.get(
            f"{self.BASE_URL}/products/{product_id}",
            headers=self._headers(),
            params=params,
            timeout=10,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("data")

    # ── Locations ─────────────────────────────────────────────────────────────

    def search_locations(self, zip_code: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find nearby Kroger store locations by zip code."""
        resp = requests.get(
            f"{self.BASE_URL}/locations",
            headers=self._headers(),
            params={
                "filter.zipCode.near": zip_code,
                "filter.limit": limit,
                "filter.radiusInMiles": 10,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    # ── User OAuth flow ──────────────────────────────────────────────────────

    def get_oauth_url(self, state: str = "") -> str:
        """
        Build the Kroger OAuth2 authorization URL.
        Redirect the user to this URL — they log in, approve scopes,
        then Kroger redirects back to KROGER_REDIRECT_URI with ?code=...
        """
        import urllib.parse
        params = {
            "client_id": settings.kroger_client_id,
            "redirect_uri": settings.kroger_redirect_uri,
            "response_type": "code",
            "scope": "openid profile email cart.basic:write product.compact",
            "state": state,
        }
        base = "https://api.kroger.com/v1/connect/oauth2/authorize"
        return f"{base}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_token(self, code: str) -> dict:
        """
        Exchange an authorization code for user access + refresh tokens.
        Call this after the user is redirected back with ?code=...
        Returns: {"access_token": ..., "refresh_token": ..., "expires_in": ...}
        """
        resp = requests.post(
            self.AUTH_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.kroger_redirect_uri,
            },
            auth=HTTPBasicAuth(settings.kroger_client_id, settings.kroger_client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def refresh_user_token(self, refresh_token: str) -> dict:
        """Exchange a refresh token for a new access token."""
        resp = requests.post(
            self.AUTH_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            auth=HTTPBasicAuth(settings.kroger_client_id, settings.kroger_client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Cart (requires user OAuth token) ─────────────────────────────────────

    def add_to_cart(
        self,
        items: list[dict],
        user_access_token: str,
    ) -> dict[str, Any]:
        """
        Add items to a Kroger user's cart.
        items format: [{"upc": "...", "quantity": 1, "modality": "PICKUP"}]
        Requires a user OAuth token with cart.basic:write scope.
        """
        resp = requests.put(
            f"{self.BASE_URL}/cart/add",
            headers={**self._headers(user_access_token), "Content-Type": "application/json"},
            json={"items": items},
            timeout=15,
        )

        if resp.status_code in (200, 204):
            return {
                "success": True,
                "order_id": resp.headers.get("X-Kroger-OrderId"),
                "total": None,  # Kroger doesn't return total here
                "error": None,
            }

        error_msg = f"Kroger cart error {resp.status_code}: {resp.text[:200]}"
        logger.error(error_msg)
        return {"success": False, "order_id": None, "total": None, "error": error_msg}
