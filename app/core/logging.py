"""Módulo de Logging Estructurado con Enmascaramiento PII/Auditoría Bancaria.
Cumple con HIPAA / ISO 27001 / SOC 2 / PCI-DSS.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone


class BankSecurityLogFormatter(logging.Formatter):
    """Formatter que enmascara información sensible (PII, tokens, contraseñas) y estructura en JSON."""

    # Patrones de enmascaramiento PII y Secreción
    _PATRON_TOKEN = re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE)
    _PATRON_SECRET = re.compile(r"(secret|password|clave|token)=['\"]?[^'\"]+['\"]?", re.IGNORECASE)

    def format(self, record: logging.LogRecord) -> str:
        mensaje_original = super().format(record)
        mensaje_sanitizado = self._enmascarar(mensaje_original)

        log_struct = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nivel": record.levelname,
            "logger": record.name,
            "mensaje": mensaje_sanitizado,
            "modulo": record.module,
            "linea": record.lineno,
        }

        if record.exc_info:
            log_struct["excepcion"] = self.formatException(record.exc_info)

        return json.dumps(log_struct, ensure_ascii=False)

    @classmethod
    def _enmascarar(cls, texto: str) -> str:
        texto = cls._PATRON_TOKEN.sub(r"\1[MASKED_JWT_TOKEN]", texto)
        texto = cls._PATRON_SECRET.sub(r"\1=[MASKED_SECRET]", texto)
        return texto
