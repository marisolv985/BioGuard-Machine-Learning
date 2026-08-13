"""Predictor baseline heurístico v0 (red de seguridad y compatibilidad v1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.schemas.prediccion import Contribucion, NivelRiesgo, PrediccionRespuesta
from app.schemas.telemetria import TelemetriaEntrada


@dataclass(frozen=True)
class RangoVital:
    etiqueta: str
    minimo_saludable: float
    maximo_saludable: float
    minimo_extremo: float
    maximo_extremo: float
    peso: float


RANGOS_VITALES: dict[str, RangoVital] = {
    "frecuencia_cardiaca": RangoVital("Frecuencia cardíaca", 60.0, 100.0, 35.0, 170.0, 0.25),
    "temperatura": RangoVital("Temperatura", 36.0, 37.5, 34.0, 42.0, 0.20),
    "saturacion_oxigeno": RangoVital("Saturación de oxígeno", 95.0, 100.0, 60.0, 100.0, 0.25),
    "frecuencia_respiratoria": RangoVital("Frecuencia respiratoria", 12.0, 20.0, 6.0, 40.0, 0.15),
    "presion_sistolica": RangoVital("Presión sistólica", 90.0, 140.0, 70.0, 200.0, 0.10),
    "presion_diastolica": RangoVital("Presión diastólica", 60.0, 90.0, 40.0, 120.0, 0.05),
    "glucosa": RangoVital("Glucosa", 70.0, 140.0, 40.0, 400.0, 0.20),
    "sudoracion_gsr": RangoVital("Sudoración (GSR)", 0.0, 4.0, 0.0, 12.0, 0.10),
    "hrv": RangoVital("HRV estimado", 40.0, 120.0, 10.0, 250.0, 0.10),
}


def _severidad(valor: float, rango: RangoVital) -> float:
    if valor < rango.minimo_saludable:
        denom = rango.minimo_saludable - rango.minimo_extremo
        if denom <= 0:
            return 1.0
        return max(0.0, min(1.0, (rango.minimo_saludable - valor) / denom))
    if valor > rango.maximo_saludable:
        denom = rango.maximo_extremo - rango.maximo_saludable
        if denom <= 0:
            return 1.0
        return max(0.0, min(1.0, (valor - rango.maximo_saludable) / denom))
    return 0.0


def _nivel_riesgo(probabilidad: float, umbral: float) -> NivelRiesgo:
    if probabilidad >= umbral:
        return NivelRiesgo.CRITICO
    if probabilidad >= 0.60:
        return NivelRiesgo.ALTO
    if probabilidad >= 0.30:
        return NivelRiesgo.MODERADO
    return NivelRiesgo.BAJO


def _scoring(telemetria: TelemetriaEntrada) -> tuple[float, float, list[Contribucion]]:
    contribuciones: list[Contribucion] = []
    severidad_ponderada = 0.0
    peso_total = 0.0
    max_severidad = 0.0

    actividad = getattr(telemetria, "actividad_fisica", None)
    fc_max_saludable = 100.0
    if actividad and str(actividad).lower() in {"activa", "activo", "intensa", "moderada", "moderado"}:
        fc_max_saludable = 135.0

    for clave, rango in RANGOS_VITALES.items():
        if clave == "sudoracion_gsr":
            # Compatibilidad con el contrato nuevo (stress_score) y el
            # campo legacy (sudoracion_gsr), pero solo si alguno fue
            # realmente enviado por el cliente.
            if (
                "stress_score" not in telemetria.model_fields_set
                and "sudoracion_gsr" not in telemetria.model_fields_set
            ):
                continue

            valor = telemetria.sudoracion_gsr_normalizado
        else:
            # Evita que valores default de campos opcionales futuros
            # alteren el scoring cuando no fueron enviados.
            if clave not in telemetria.model_fields_set:
                continue

            valor = getattr(telemetria, clave, None)

        if valor is None:
            continue
        valor_flt = float(valor)

        if clave == "frecuencia_cardiaca" and fc_max_saludable > 100.0:
            rango = RangoVital(
                rango.etiqueta,
                rango.minimo_saludable,
                fc_max_saludable,
                rango.minimo_extremo,
                185.0,
                rango.peso,
            )

        severidad = _severidad(valor_flt, rango)

        if clave == "glucosa" and getattr(telemetria, "toma_reciente_medicamento", False) is True:
            severidad *= 0.75

        contribuciones.append(
            Contribucion(
                senal=rango.etiqueta,
                valor=round(valor_flt, 4),
                severidad=round(severidad, 4),
            )
        )
        severidad_ponderada += rango.peso * severidad
        peso_total += rango.peso
        max_severidad = max(max_severidad, severidad)

    if peso_total <= 0:
        raise ValueError("no se recibieron señales vitales suficientes para calcular el riesgo")

    severidad_media = severidad_ponderada / peso_total
    return severidad_media, max_severidad, contribuciones


def predecir_baseline(
    telemetria: TelemetriaEntrada,
    settings: Any,
) -> PrediccionRespuesta:
    """Probabilidad de crisis metabólica (heurística explicable, determinista)."""
    severidad_media, max_severidad, contribuciones = _scoring(telemetria)
    peso_peor = settings.baseline_peso_peor_senal
    efectiva = peso_peor * max_severidad + (1.0 - peso_peor) * severidad_media
    logit = 6.0 * efectiva - 3.0
    probabilidad = 1.0 / (1.0 + math.exp(-logit))
    probabilidad = round(probabilidad, 4)
    es_critico = probabilidad >= settings.umbral_critico

    notas_explicacion = [
        "Baseline v0: severidad por desviación de rangos vitales "
        "(peor señal + media ponderada) convertida a probabilidad logística."
    ]
    if getattr(telemetria, "toma_reciente_medicamento", False) is True:
        notas_explicacion.append("Toma reciente de medicamento detectada: riesgo atenuado en glucosa.")
    if getattr(telemetria, "actividad_fisica", None):
        act = telemetria.actividad_fisica
        notas_explicacion.append(f"Ajuste por actividad física ({act}) aplicado en umbrales de FC.")

    return PrediccionRespuesta(
        paciente_id=telemetria.paciente_id,
        probabilidad=probabilidad,
        es_critico=es_critico,
        nivel_riesgo=_nivel_riesgo(probabilidad, settings.umbral_critico),
        umbral_critico=settings.umbral_critico,
        timestamp=datetime.now(timezone.utc),
        modelo_id="baseline-v0",
        version=settings.app_version,
        mensaje=(
            "ALERTA: probabilidad de crisis metabólica por encima del umbral crítico"
            if es_critico
            else "Estado estable: probabilidad por debajo del umbral crítico"
        ),
        contribuciones=contribuciones,
        explicacion=" ".join(notas_explicacion),
    )
