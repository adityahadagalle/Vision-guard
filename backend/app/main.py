"""VisionGuard FastAPI application entry point.

Initializes the FastAPI application, registers route handlers, configures logging,
adds secure structured exception handling, and reconciles background job states.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.api.v1.router import api_v1_router
from backend.app.api.v1.routes.distribution_shift import (
    router as distribution_shift_router,
)
from backend.app.api.v1.routes.health import router as root_health_router
from backend.app.api.v1.routes.inference import router as inference_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import setup_logging
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.db.repositories.upload_reservation_repository import (
    SqliteUploadReservationRepository,
)
from backend.app.db.session import get_session_factory

logger = logging.getLogger("visionguard.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling startup initialization and shutdown cleanup."""
    # Startup: reconcile stranded RUNNING jobs and stale upload reservations from crashes
    try:
        session = get_session_factory()()
        try:
            queue = SqliteJobQueue(session)
            recovered = queue.recover_strands_on_startup()
            if recovered > 0:
                logger.warning(
                    "Startup job recovery: reconciled %d stranded RUNNING job(s).", recovered
                )

            model_repo = SqliteModelRepository(session)
            reconciled_models = model_repo.reconcile_stranded_received_models(queue)
            if reconciled_models > 0:
                logger.warning(
                    "Startup model recovery: reconciled %d stranded RECEIVED model(s).",
                    reconciled_models,
                )

            res_repo = SqliteUploadReservationRepository(session)
            settings = get_settings()
            quarantine_dir = settings.data_dir / "quarantine"
            stale_cleaned = res_repo.reconcile_and_cleanup_stale(quarantine_dir)
            if stale_cleaned > 0:
                logger.warning(
                    "Startup quarantine recovery: cleaned %d stale/expired upload reservation(s).",
                    stale_cleaned,
                )
        finally:
            session.close()
    except Exception as err:
        logger.error("Failed to run startup reconciliation: %s", err)

    yield


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """Application factory for VisionGuard.

    Args:
        custom_settings: Optional custom Settings instance for testing/override.

    Returns:
        Configured FastAPI application instance.
    """
    settings = custom_settings or get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "VisionGuard is an offline-first AI lifecycle integrity and security "
            "assurance platform for computer-vision systems."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 1. Custom exception handlers: defensive error responses without path or stack leakage
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            content = exc.detail
        else:
            content = {"error_code": "HTTP_ERROR", "message": str(exc.detail), "details": {}}
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        sanitized_errors = [
            {
                "type": err.get("type", "value_error"),
                "loc": [str(x) for x in err.get("loc", [])],
                "msg": str(err.get("msg", "")),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error_code": "VALIDATION_ERROR",
                "message": "Request payload validation failed.",
                "details": {"errors": sanitized_errors},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
                "details": {},
            },
        )

    # Route registrations:
    # 1. Mount health router at root for direct /health monitoring probes
    app.include_router(root_health_router)

    # 2. Mount versioned API routes under /api/v1
    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
    app.include_router(inference_router, prefix=settings.api_v1_prefix)
    app.include_router(distribution_shift_router, prefix=settings.api_v1_prefix)

    if custom_settings is not None:
        app.dependency_overrides[get_settings] = lambda: custom_settings

    return app


# Default ASGI application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    current_settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=current_settings.host,
        port=current_settings.port,
        reload=current_settings.env == "development",
    )
