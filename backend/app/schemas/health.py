"""Health check schema."""

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Structured response model for application health check endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "status": "healthy",
                "application": "VisionGuard",
                "version": "0.1.0",
            }
        },
    )

    status: str = Field(
        default="healthy",
        description="Current application health status",
    )
    application: str = Field(
        default="VisionGuard",
        description="Name of the running application",
    )
    version: str = Field(
        default="0.1.0",
        description="Semantic version of the application",
    )
