"""Langfuse trace provider configuration."""

from __future__ import annotations

import base64

from livekit import agents
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from src.core.logger import logger
from src.core.settings import settings

_langfuse_tracer_provider: TracerProvider | None = None


def _normalize_langfuse_host() -> str | None:
    host = settings.langfuse.LANGFUSE_HOST or settings.langfuse.LANGFUSE_BASE_URL
    if not host:
        return None
    return host.rstrip("/")


def setup_langfuse_tracer() -> TracerProvider | None:
    """Configure LiveKit telemetry tracer to export traces to Langfuse."""
    global _langfuse_tracer_provider

    if not settings.langfuse.LANGFUSE_ENABLED:
        return None
    if _langfuse_tracer_provider is not None:
        return _langfuse_tracer_provider

    host = _normalize_langfuse_host()
    public_key = settings.langfuse.LANGFUSE_PUBLIC_KEY
    secret_key = settings.langfuse.LANGFUSE_SECRET_KEY
    if not host or not public_key or not secret_key:
        logger.warning(
            "Langfuse tracing enabled but LANGFUSE_HOST/LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are missing"
        )
        return None

    try:
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("utf-8")
        span_exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )
        tracer_provider = TracerProvider(
            resource=Resource.create(
                {
                    SERVICE_NAME: "open-voice-agent",
                    SERVICE_VERSION: getattr(agents, "__version__", "unknown"),
                    "deployment.environment": settings.langfuse.LANGFUSE_ENVIRONMENT,
                }
            )
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        set_tracer_provider(tracer_provider)
        _langfuse_tracer_provider = tracer_provider
        logger.info("Langfuse OTEL tracing configured")
        return tracer_provider
    except Exception as exc:
        logger.warning(f"Failed to set up Langfuse tracing: {exc}")
        return None
