"""Common API response models."""

import time
from typing import Any

from pydantic import BaseModel, Field

_API_VERSION: str = "0.4.0"


class Meta(BaseModel):
    """Response metadata."""

    timestamp: float = Field(default_factory=time.time)
    version: str = _API_VERSION


class ResponseEnvelope(BaseModel):
    """Standard API response envelope.

    All API responses are wrapped in this envelope:
    {
        "status": "success" | "error",
        "data": { ... },
        "meta": { "timestamp": ..., "version": "0.4.0" }
    }
    """

    status: str = "success"
    data: Any = None
    meta: Meta = Field(default_factory=Meta)


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str
    message: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    status: str = "error"
    error: ErrorDetail
    meta: Meta = Field(default_factory=Meta)


def success_response(data: Any = None) -> dict:
    """Create a success response envelope."""
    return ResponseEnvelope(status="success", data=data).model_dump()


def error_response(code: str, message: str, detail: str | None = None) -> dict:
    """Create an error response envelope."""
    return ErrorResponse(
        error=ErrorDetail(code=code, message=message, detail=detail),
    ).model_dump()
