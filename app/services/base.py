from abc import ABC, abstractmethod

from app.schemas.prediccion import NivelRiesgo, PrediccionRespuesta
from app.schemas.telemetria import TelemetriaEntrada


def _nivel_riesgo(probabilidad: float, umbral: float) -> NivelRiesgo:
    if probabilidad >= umbral:
        return NivelRiesgo.CRITICO
    if probabilidad >= 0.60:
        return NivelRiesgo.ALTO
    if probabilidad >= 0.30:
        return NivelRiesgo.MODERADO
    return NivelRiesgo.BAJO


class PredictorBase(ABC):
    @abstractmethod
    async def predecir(self, telemetria: TelemetriaEntrada) -> PrediccionRespuesta:
        raise NotImplementedError
