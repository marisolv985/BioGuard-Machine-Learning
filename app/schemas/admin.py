from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import SALIDA_CFG, ENTRADA_CFG


class EntrenarRequest(BaseModel):
    model_config = ENTRADA_CFG

    descripcion: Optional[str] = Field(default=None, max_length=500)
    retener_activo: bool = True


class ModeloRegistro(BaseModel):
    model_config = SALIDA_CFG

    id: str
    version: str
    tipo: str
    activo: bool
    total_muestras: int
    fecha_entrenamiento: Optional[datetime] = None
    descripcion: Optional[str] = None
    metricas: Optional[dict] = None
    conteos: Optional[dict] = None


class ModeloActivoResponse(BaseModel):
    model_config = SALIDA_CFG

    version: str
    tipo: str
    activo: bool
    metricas: Optional[dict] = None
    total_muestras: int
    fecha_entrenamiento: Optional[datetime] = None
    features: list[str] = []
    drift: Optional[dict] = None
