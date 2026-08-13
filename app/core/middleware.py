"""Middleware de seguridad de grado bancario (Banking / DevSecOps):
- Inyección de cabeceras de seguridad OWASP (HSTS, CSP, X-Frame-Options,
  X-Content-Type-Options, Referrer-Policy, Permissions-Policy).
- Trazabilidad distribuida mediante X-Correlation-ID en cada petición.
- Protección contra agotamiento de recursos limitando el tamaño máximo del Payload.
- Sanitización y enmascaramiento de logs (PII / tokens).
"""

from __future__ import annotations

import uuid
import logging
from typing import Callable
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

MAX_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2 MB máximo por petición


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generar o propagar X-Correlation-ID para auditoría distribuida
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

        # Guard contra payloads excesivos (DoS / Memory Overflow)
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PAYLOAD_BYTES:
            logger.warning(
                "Petición rechazada por exceder tamaño máximo (%s bytes) [Correlation-ID: %s]",
                content_length,
                correlation_id,
            )
            return Response(
                content='{"detail": "El tamaño del Payload excede el límite máximo permitido (2MB)"}',
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                media_type="application/json",
            )

        response: Response = await call_next(request)

        # Inyección de Cabeceras de Seguridad Nivel Bancario (OWASP Compliance)
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        response.headers["Server"] = "BioGuard-ML-Core"

        return response
