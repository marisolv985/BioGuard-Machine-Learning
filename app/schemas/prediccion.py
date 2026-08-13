from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import SALIDA_CFG


class NivelRiesgo(str, Enum):
    BAJO = "BAJO"
    MODERADO = "MODERADO"
    ALTO = "ALTO"
    CRITICO = "CRITICO"


class TipoRiesgo(str, Enum):
    NORMAL = "normal"
    RIESGO_HIPO = "riesgo_hipo"
    RIESGO_HIPER = "riesgo_hiper"


class Contribucion(BaseModel):
    model_config = SALIDA_CFG

    senal: str = Field(..., description="Nombre legible de la señal vital")
    valor: float = Field(...)
    severidad: float = Field(..., ge=0.0, le=1.0)


# ---------- v1 (retrocompatible con el contrato anterior) ----------


class PrediccionRespuesta(BaseModel):
    model_config = SALIDA_CFG

    paciente_id: str
    probabilidad: float = Field(ge=0.0, le=1.0)
    es_critico: bool
    nivel_riesgo: NivelRiesgo
    umbral_critico: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    modelo_id: str
    version: str
    mensaje: Optional[str] = None
    contribuciones: Optional[list[Contribucion]] = None
    explicacion: Optional[str] = None


# ---------- v2 (orientada a dashboard y distinción hipo/hiper) ----------


class RiesgoActual(BaseModel):
    model_config = SALIDA_CFG

    tipo: TipoRiesgo
    probabilidad: float = Field(ge=0.0, le=1.0)
    probabilidad_hipo: float = Field(ge=0.0, le=1.0)
    probabilidad_hiper: float = Field(ge=0.0, le=1.0)
    umbral_critico: float = Field(ge=0.0, le=1.0)
    horas_estimadas: int = Field(ge=0)
    es_critico: bool


class AporteExplicativo(BaseModel):
    model_config = SALIDA_CFG

    senal: str
    aporte: float
    detalle: str = ""


class ModeloInfo(BaseModel):
    model_config = SALIDA_CFG

    id: str
    version: str
    activo: bool


class PrediccionV2(BaseModel):
    model_config = SALIDA_CFG

    paciente_id: str
    riesgo: RiesgoActual
    explicacion: list[AporteExplicativo] = []
    recomendacion: str = ""
    modelo: ModeloInfo
    timestamp: datetime


# ---------- v3 (picos glucémicos: fórmulas F1-F3 + matriz de riesgo) ----------


class PicoGlucemicoRespuesta(BaseModel):
    """Salida exacta del motor de picos glucémicos (F1/F2/F3 + matriz).

    Campos de negocio: imc, z, pPico, casoClinico, accionAutomatizada.
    """

    model_config = SALIDA_CFG

    paciente_id: str
    imc: float = Field(ge=0.0, le=120.0)
    z: float
    p_pico: float = Field(ge=0.0, le=1.0)
    caso_clinico: str
    nivel_riesgo: str = ""
    accion_automatizada: str
    timestamp: datetime
    version: str
