"""
api/kroger.py
Kroger API client — handles OAuth2 client credentials flow (for product search)
and user-delegated auth (for cart operations).

Fixes applied:
  - Exponential backoff + retry on 429 (rate limit) and 5xx errors
  - Token auto-refresh on 401 without leaking stale tokens
  - Better error messages surfaced (not just status codes)
  - Request timeout increased for slow Kroger sandbox responses
  - Cart add_to_cart returns structured error details per item
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from core.config import settings

logger = logging.getLogger(__name__)

# Max retries for transient errors
MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}


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

    def invalidate(self) -> None:
        self._token = None
        self._expires_at = 0.0


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

        resp = self._post_with_retry(
            self.AUTH_URL,
            data={"grant_type": "client_credentials", "scope": "product.compact"},
            auth=HTTPBasicAuth(settings.kroger_client_id, settings.kroger_client_secret),
        )
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

    # ── Retry helpers ─────────────────────────────────────────────────────────

    def _get_with_retry(self, url: str, **kwargs) -> requests.Response:
        """GET with exponential backoff on rate limit / server errors."""
        kwargs.setdefault("timeout", 20)
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, **kwargs)
                if resp.status_code == 401:
                    # Token expired — invalidate and refresh once
                    _app_token_cache.invalidate()
                    kwargs["headers"] = self._headers()
                    resp = requests.get(url, **kwargs)
                if resp.status_code in RETRY_STATUSES:
                    wait = 2 ** attempt
                    logger.warning(
                        "Kroger GET %s returned %d — retry %d/%d in %ds",
                        url, resp.status_code, attempt + 1, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("Kroger GET failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)
                time.sleep(wait)
        raise last_exc or RuntimeError("Kroger GET failed after retries")

    def _post_with_retry(self, url: str, **kwargs) -> requests.Response:
        """POST with exponential backoff."""
        kwargs.setdefault("timeout", 15)
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(url, **kwargs)
                if resp.status_code in RETRY_STATUSES:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise last_exc or RuntimeError("Kroger POST failed after retries")

    # ── Product Search ────────────────────────────────────────────────────────

    def search_products(
        self,
        query: str,
        location_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for products by keyword.
        location_id is optional — omitting it works with sandbox credentials
        but won't return pricing (prices require a production app with location access).
        """
        params: dict[str, Any] = {
            "filter.term": query,
            "filter.limit": limit,
        }
        if location_id:
            params["filter.locationId"] = location_id

        try:
            resp = self._get_with_retry(
                f"{self.BASE_URL}/products",
                headers=self._headers(),
                params=params,
            )
            return resp.json().get("data", [])
        except Exception as exc:
            logger.error("Product search failed for '%s': %s", query, exc)
            return []

    def get_product(self, product_id: str, location_id: str = "") -> dict[str, Any] | None:
        """Fetch a single product by ID."""
        params = {}
        if location_id:
            params["filter.locationId"] = location_id

        try:
            resp = self._get_with_retry(
                f"{self.BASE_URL}/products/{product_id}",
                headers=self._headers(),
                params=params,
            )
            return resp.json().get("data")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    # ── Locations ─────────────────────────────────────────────────────────────

    def search_locations(self, zip_code: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find nearby Kroger store locations by zip code."""
        try:
            resp = self._get_with_retry(
                f"{self.BASE_URL}/locations",
                headers=self._headers(),
                params={
                    "filter.zipCode.near": zip_code,
                    "filter.limit": limit,
                    "filter.radiusInMiles": 10,
                },
            )
            return resp.json().get("data", [])
        except Exception as exc:
            logger.error("Location search failed for zip '%s': %s", zip_code, exc)
            return []

    # ── User OAuth flow ───────────────────────────────────────────────────────

    def get_oauth_url(self, state: str = "") -> str:
        """Build the Kroger OAuth2 authorization URL."""
        import urllib.parse
        params = {
            "client_id": settings.kroger_client_id,
            "redirect_uri": settings.kroger_redirect_uri,
            "response_type": "code",
            "scope": "cart.basic:write product.compact",
            "state": state,
        }
        base = "https://api.kroger.com/v1/connect/oauth2/authorize"
        return f"{base}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange an authorization code for user access + refresh tokens."""
        resp = self._post_with_retry(
            self.AUTH_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.kroger_redirect_uri,
            },
            auth=HTTPBasicAuth(settings.kroger_client_id, settings.kroger_client_secret),
        )
        return resp.json()

    def refresh_user_token(self, refresh_token: str) -> dict:
        """Exchange a refresh token for a new access token."""
        resp = self._post_with_retry(
            self.AUTH_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            auth=HTTPBasicAuth(settings.kroger_client_id, settings.kroger_client_secret),
        )
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

        Returns structured result with success, order_id, and error details.
        """
        try:
            resp = requests.put(
                f"{self.BASE_URL}/cart/add",
                headers={
                    **self._headers(user_access_token),
                    "Content-Type": "application/json",
                },
                json={"items": items},
                timeout=20,
            )

            if resp.status_code in (200, 204):
                return {
                    "success": True,
                    "order_id": resp.headers.get("X-Kroger-OrderId"),
                    "total": None,
                    "error": None,
                }

            # Parse Kroger error body for a better message
            try:
                error_body = resp.json()
                error_msg = error_body.get("errors", [{}])[0].get("reason", resp.text[:200])
            except Exception:
                error_msg = resp.text[:200]

            if resp.status_code == 401:
                error_msg = "Kroger session expired. Please log in again."
            elif resp.status_code == 403:
                error_msg = "Cart access denied. Ensure cart.basic:write scope is approved."

            logger.error("Kroger cart error %d: %s", resp.status_code, error_msg)
            return {"success": False, "order_id": None, "total": None, "error": error_msg}

        except requests.RequestException as exc:
            logger.error("Kroger cart request failed: %s", exc)
            return {
                "success": False,
                "order_id": None,
                "total": None,
                "error": f"Network error adding to cart: {exc}",
            }
