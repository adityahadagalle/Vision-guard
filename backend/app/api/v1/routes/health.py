"""Health check endpoint route."""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application Health Check",
    description="Reports application process health only. Does not assert that ML modules are operational.",
)
async def get_health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Return application health status, name, and version."""
    return HealthResponse(
        status="healthy",
        application=settings.app_name,
        version=settings.app_version,
    )
