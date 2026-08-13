from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings

_settings = get_settings()

# Rate limiting: se activa solo en producción para no entorpecer las pruebas locales.
limiter = Limiter(key_func=get_remote_address, enabled=_settings.is_production)
