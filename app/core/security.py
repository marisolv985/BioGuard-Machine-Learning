from datetime import datetime, timezone

import jwt
from fastapi import Header, HTTPException, status
from jwt import InvalidTokenError

from app.core.config import Settings, get_settings

_BEARER = "Bearer "


def _extraer_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.startswith(_BEARER):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cabecera Authorization ausente o con formato inválido",
        )
    return authorization[len(_BEARER) :].strip()


def crear_token_servicio(settings: Settings, ttl_seg: int = 120) -> str:
    """Genera un token JWT de servicio (uso interno y pruebas)."""
    now = datetime.now(timezone.utc)
    payload = {
        "iss": settings.service_issuer,
        "aud": settings.service_audience,
        "tipo": "servicio",
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + ttl_seg,
    }
    return jwt.encode(payload, settings.service_token_secret, algorithm="HS256")


def verificar_token_servicio(
    authorization: str | None = Header(default=None),
    settings: Settings = get_settings(),
) -> None:
    """Dependencia FastAPI: exige un JWT de servicio-a-servicio firmado por el backend .NET."""
    token = _extraer_bearer(authorization)
    try:
        payload = jwt.decode(
            token,
            settings.service_token_secret,
            algorithms=["HS256"],
            audience=settings.service_audience,
            issuer=settings.service_issuer,
            options={"require": ["exp", "iat"]},
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de servicio inválido o expirado",
        ) from exc
    if payload.get("tipo") != "servicio":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El token no es de tipo servicio",
        )
