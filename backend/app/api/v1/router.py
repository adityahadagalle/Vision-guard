"""API v1 master router aggregating sub-routes."""

from fastapi import APIRouter

from backend.app.api.v1.routes.assessments import router as assessments_router
from backend.app.api.v1.routes.datasets import router as datasets_router
from backend.app.api.v1.routes.health import router as health_router
from backend.app.api.v1.routes.jobs import router as jobs_router
from backend.app.api.v1.routes.models import router as models_router

api_v1_router = APIRouter()

# Register sub-routers under /api/v1
api_v1_router.include_router(health_router)
api_v1_router.include_router(datasets_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(assessments_router)
api_v1_router.include_router(jobs_router)
