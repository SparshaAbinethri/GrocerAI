import logging
import os
import time
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from core.logging_config import setup_logging, get_logger
from core.metrics import setup_metrics
from core.security import limiter, rate_limit_handler, SecurityHeadersMiddleware

# ---------------------------------------------------------------------------
# Logging — must be set up before anything else
# ---------------------------------------------------------------------------
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentry — error tracking
# Set SENTRY_DSN in Railway environment variables
# ---------------------------------------------------------------------------
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.2")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
        environment=os.getenv("ENVIRONMENT", "production"),
        release=os.getenv("APP_VERSION", "1.0.0"),
        send_default_pii=False,
    )
    logger.info("Sentry initialized", extra={"environment": os.getenv("ENVIRONMENT")})
else:
    logger.warning("SENTRY_DSN not set — error tracking is disabled")


# ---------------------------------------------------------------------------
# App lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Application starting up",
        extra={
            "version": os.getenv("APP_VERSION", "1.0.0"),
            "environment": os.getenv("ENVIRONMENT", "production"),
        },
    )
    yield
    logger.info("Application shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

app = FastAPI(
    title="Your API",
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan,
    # Hide Swagger/ReDoc in production — set ENVIRONMENT=development locally
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if ENVIRONMENT != "production" else None,
)

# ---------------------------------------------------------------------------
# Middlewares — order matters (outermost = last added)
# ---------------------------------------------------------------------------
app.add_middleware(SecurityHeadersMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    logger.info(
        "Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
        },
    )

    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    # Add response time header for debugging
    response.headers["X-Response-Time"] = f"{duration_ms}ms"
    return response


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
setup_metrics(app)


# ---------------------------------------------------------------------------
# Global exception handler — catches unhandled errors, logs them
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        extra={
            "method": request.method,
            "path": request.url.path,
            "error": str(exc),
            "error_type": type(exc).__name__,
        },
        exc_info=True,
    )
    # Sentry captures this automatically via sentry_sdk.init()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Health & Readiness endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Observability"], include_in_schema=True)
async def health_check():
    """
    Liveness probe — confirms the app is running.
    Railway uses this to detect crashed containers.
    """
    return {
        "status": "healthy",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "environment": ENVIRONMENT,
    }


@app.get("/ready", tags=["Observability"], include_in_schema=True)
async def readiness_check():
    """
    Readiness probe — confirms the app is ready to serve traffic.
    Add DB/cache connectivity checks here.
    """
    checks = {}

    # Example: check database connection
    # try:
    #     await db.execute("SELECT 1")
    #     checks["database"] = "ok"
    # except Exception as e:
    #     checks["database"] = f"error: {e}"
    #     return JSONResponse(status_code=503, content={"status": "not ready", "checks": checks})

    checks["app"] = "ok"
    return {"status": "ready", "checks": checks}


# ---------------------------------------------------------------------------
# Your existing routes — plug them in here
# ---------------------------------------------------------------------------
# from routers import users, products
# app.include_router(users.router, prefix="/api/v1")
# app.include_router(products.router, prefix="/api/v1")
