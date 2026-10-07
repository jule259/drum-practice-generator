"""FastAPI application factory and HTTP routes."""

from __future__ import annotations

from .app import create_app  # noqa: F401

__all__ = ["create_app"]
