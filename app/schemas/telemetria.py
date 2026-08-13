from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ENTRADA_CFG


class TelemetriaEntrada(BaseModel):
    """Lectura fisiológica entrante (post /predicciones). Compatible con v1."""

    model_config = ENTRADA_CFG

    paciente_id: str = Field(..., min_length=1, max_length=64, description="Identificador del paciente")
    frecuencia_cardiaca: int = Field(..., ge=20, le=250, description="Pulsaciones por minuto")
    temperatura: float = Field(..., ge=30.0, le=45.0, description="Temperatura corporal en °C")
    saturacion_oxigeno: float = Field(..., ge=50.0, le=100.0, description="SpO2 en %")
    frecuencia_respiratoria: int = Field(..., ge=4, le=60, description="Respiraciones por minuto")
    presion_sistolica: Optional[int] = Field(default=None, ge=50, le=260)
    presion_diastolica: Optional[int] = Field(default=None, ge=30, le=160)
    glucosa: Optional[float] = Field(default=None, ge=20.0, le=600.0, description="Glucosa en mg/dL")
    stress_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Nivel de estrés normalizado 0-100 (contrato unificado)"
    )
    sudoracion_gsr: Optional[float] = Field(default=None, ge=0.0, le=20.0, description="Sudoración GSR (legacy)")
    hrv: Optional[float] = Field(default=None, ge=10.0, le=200.0, description="HRV estimado en ms")
    peso: Optional[float] = Field(default=None, ge=20.0, le=400.0, description="Peso en kg (para IMC)")
    estatura: Optional[float] = Field(
        default=None, ge=0.5, le=2.5, description="Estatura en metros (para IMC)"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dispositivo: Optional[str] = Field(default=None, max_length=64)
    toma_reciente_medicamento: Optional[bool] = Field(default=None, description="Toma de medicamento")
    es_sueno: Optional[bool] = Field(default=None, description="Horario de sueño")
    actividad_fisica: Optional[str] = Field(default=None, description="Nivel de actividad física")

    @model_validator(mode="after")
    def _validar_presiones(self) -> "TelemetriaEntrada":
        if (
            self.presion_sistolica is not None
            and self.presion_diastolica is not None
            and self.presion_sistolica <= self.presion_diastolica
        ):
            raise ValueError("la presión sistólica debe ser mayor que la diastólica")
        return self

    @property
    def sudoracion_gsr_normalizado(self) -> float:
        """Estrés 0-100 (contrato nuevo) mapeado al rango GSR 0-20 que usa el modelo."""
        if self.stress_score is not None:
            return self.stress_score / 5.0
        return float(self.sudoracion_gsr or 0.0)
