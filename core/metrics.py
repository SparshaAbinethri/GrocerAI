from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI


def setup_metrics(app: FastAPI) -> None:
    """
    Expose Prometheus metrics at /metrics endpoint.

    Tracks:
    - Request count by method, path, status code
    - Request duration (latency histogram)
    - Requests in progress
    - Default Python/process metrics
    """
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", "/ready", "/metrics"],
        inprogress_name="fastapi_requests_inprogress",
        inprogress_labels=True,
    ).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,  # Hide from Swagger docs
    )
    
import logging
logger = logging.getLogger(__name__)

def record_pipeline_run(success: bool, duration_seconds: float, **kwargs):
    """Record a pipeline run result."""
    logger.info(
        "Pipeline run recorded",
        extra={
            "success": success,
            "duration_seconds": round(duration_seconds, 3),
            **kwargs
        }
    )