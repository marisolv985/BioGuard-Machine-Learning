from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import SALIDA_CFG, ENTRADA_CFG
from app.schemas.prediccion import AporteExplicativo, ModeloInfo, RiesgoActual


class RangoTendencia(str, Enum):
    DIA = "dia"
    SEMANA = "semana"
    MES = "mes"


class PuntoTendencia(BaseModel):
    model_config = SALIDA_CFG

    fecha: datetime
    riesgo_promedio: float = Field(ge=0.0, le=1.0)
    probabilidad_pico: float = Field(ge=0.0, le=1.0)
    n_predicciones: int = Field(ge=0)


class ResumenDashboard(BaseModel):
    model_config = SALIDA_CFG

    paciente_id: str
    riesgo_actual: Optional[RiesgoActual] = None
    explicacion: list[AporteExplicativo] = []
    recomendacion: str = ""
    tendencia: list[PuntoTendencia] = []
    modelo: Optional[ModeloInfo] = None
    ultima_actualizacion: datetime


class HistorialItem(BaseModel):
    model_config = SALIDA_CFG

    prediccion_id: str
    probabilidad: float = Field(ge=0.0, le=1.0)
    tipo: str
    nivel_riesgo: str
    timestamp: datetime
    es_critico: bool


class HistorialPage(BaseModel):
    model_config = SALIDA_CFG

    items: list[HistorialItem]
    total: int
    pagina: int = Field(ge=1)
    pagina_tamano: int = Field(ge=1, le=100)


class EventoConfirmacionRequest(BaseModel):
    model_config = ENTRADA_CFG

    paciente_id: str = Field(..., min_length=1, max_length=64)
    evento_id: str = Field(..., min_length=1, max_length=64)
    confirmado: bool
    nota: Optional[str] = Field(default=None, max_length=500)
