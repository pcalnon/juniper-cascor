"""Common API response models."""

import time
from typing import Any

from pydantic import BaseModel, Field

_API_VERSION: str = "0.6.0"


def coerce_native_scalars(value: Any) -> Any:
    """Coerce numpy scalar types to Python natives, recursively.

    pydantic-core's JSON serializer rejects ``numpy.int64`` /
    ``numpy.float64`` with ``PydanticSerializationError: Unable to
    serialize unknown type: <class 'numpy.int64'>``. After
    ``load_snapshot`` (and other code paths that read scalars back
    from h5py / numpy arrays), the network's scalar attributes come
    back as numpy scalars rather than Python natives. Applied at the
    response envelope so every route's payload is JSON-clean
    regardless of whether the route author knew to coerce per-field.

    Walks dicts and lists/tuples; passes through anything that
    doesn't expose ``.item()``. Plain Python scalars (str/int/float/
    bool/None) don't have ``.item()`` so they're untouched.
    """
    if isinstance(value, dict):
        return {k: coerce_native_scalars(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        coerced = [coerce_native_scalars(v) for v in value]
        return type(value)(coerced) if isinstance(value, tuple) else coerced
    item = getattr(value, "item", None)
    if callable(item):
        # numpy scalars and 0-d numpy arrays expose ``.item()``
        # returning a Python native.
        try:
            return item()
        except (ValueError, TypeError):
            # Defensive: any object whose ``.item()`` doesn't behave
            # like a numpy scalar's (e.g. takes args, raises) falls
            # through unchanged.
            return value
    return value


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
        "meta": { "timestamp": ..., "version": "0.6.0" }
    }
    """

    status: str = "success"
    data: Any = None
    meta: Meta = Field(default_factory=Meta)


class ErrorDetail(BaseModel):
    """Error detail model.

    ``detail`` carries a *structured* payload when one exists, and stays
    ``None`` otherwise. Two shapes are in use and both are deliberate:

    * ``None`` -- every ``HTTPException`` route. The human-readable text is
      ``message``; there is no per-field structure to carry. Pinned by
      ``test_api_09_http_exception_envelope.py``.
    * ``list[dict]`` -- request-validation failures (422). FastAPI's
      ``RequestValidationError.errors()`` is a list of per-field objects
      (``{"type", "loc", "msg", ...}``), and that structure is the whole value
      of a validation error: it says *which field* failed and *why*.
      Flattening it into a string is not an acceptable simplification --
      ``juniper-cascor-client``'s ``_render_error_detail`` renders it and
      ``exc.detail`` exposes it unmodified, so collapsing it here would
      destroy information the published client is built to consume. This is
      the same trap ``juniper-data`` recorded on ``APD-DATA-013``.

    ``str`` remains accepted for callers that pass prose explicitly.
    """

    code: str
    message: str
    detail: str | list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    status: str = "error"
    error: ErrorDetail
    meta: Meta = Field(default_factory=Meta)


def success_response(data: Any = None) -> dict:
    """Create a success response envelope.

    ``data`` is run through ``coerce_native_scalars`` so any
    numpy.int64 / numpy.float64 / 0-d ndarray scalars upstream
    (typically from h5py-deserialized network attributes after a
    snapshot restore) become Python natives before pydantic-core
    serializes the envelope to JSON. Without this, those endpoints
    return 400 ``VALIDATION_ERROR`` with a stripped detail because
    ``PydanticSerializationError`` is a ``ValueError`` subclass and
    the cascor app's ``value_error_handler`` catches it.
    """
    return ResponseEnvelope(status="success", data=coerce_native_scalars(data)).model_dump()


def error_response(code: str, message: str, detail: "str | list[dict[str, Any]] | None" = None) -> dict:
    """Create an error response envelope.

    ``detail`` accepts the structured ``list[dict]`` that
    ``RequestValidationError.errors()`` produces as well as prose; see
    :class:`ErrorDetail` for why the list must not be flattened.
    """
    return ErrorResponse(
        error=ErrorDetail(code=code, message=message, detail=detail),
    ).model_dump()
