"""Inferencia v2: distinción hipo/hiper, explicabilidad local y fallback heurístico."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.models.features import FEATURES, FEATURES_LABELS, features_vector

_CLASE_HIPO = "riesgo_hipo"
_CLASE_HIPER = "riesgo_hiper"

RECOMENDACIONES: dict[str, str] = {
    "normal": "Estado estable. Sin recomendación urgente.",
    _CLASE_HIPO: (
        "Verifica síntomas de hipoglucemia. "
        "Considera ingerir una fuente de azúcar rápida y monitoriza."
    ),
    _CLASE_HIPER: "Posible hiperglucemia. Evita azúcares, verifica medicación registrada e hidrátate.",
}


class ModeloActivo:
    """Wrapper sobre el artefacto entrenado (serializado) para inferencia."""

    def __init__(self, artefacto: dict[str, Any] | None, umbral: float) -> None:
        self.artefacto = artefacto
        self.umbral = umbral

    @property
    def disponible(self) -> bool:
        return self.artefacto is not None and (
            self.artefacto.get("modelo_hipo") is not None or self.artefacto.get("modelo_hiper") is not None
        )

    @property
    def version(self) -> str | None:
        return self.artefacto.get("version") if self.artefacto else None

    def _probas(self, x_vec: np.ndarray) -> tuple[float, float]:
        if not self.artefacto:
            return 0.0, 0.0
        imputer = self.artefacto["imputer"]
        x = imputer.transform(x_vec.reshape(1, -1))
        p_hipo = 0.0
        p_hiper = 0.0
        m_hipo = self.artefacto.get("modelo_hipo")
        m_hiper = self.artefacto.get("modelo_hiper")
        if m_hipo is not None:
            p_hipo = float(m_hipo.predict_proba(x)[0, 1])
        if m_hiper is not None:
            p_hiper = float(m_hiper.predict_proba(x)[0, 1])
        return p_hipo, p_hiper

    def predecir(self, features: dict[str, float]) -> dict[str, Any]:
        """Devuelve {tipo, probabilidad, probabilidad_hipo, probabilidad_hiper, es_critico}."""
        x_vec = features_vector(features)
        p_hipo, p_hiper = self._probas(x_vec)
        p_normal = max(0.0, 1.0 - p_hipo - p_hiper)
        if p_hipo >= p_hiper and p_hipo > p_normal:
            tipo = _CLASE_HIPO
            probabilidad = p_hipo
        elif p_hiper > p_normal:
            tipo = _CLASE_HIPER
            probabilidad = p_hiper
        else:
            tipo = "normal"
            probabilidad = p_normal
        return {
            "tipo": tipo,
            "probabilidad": float(probabilidad),
            "probabilidad_hipo": p_hipo,
            "probabilidad_hiper": p_hiper,
            "es_critico": bool(max(p_hipo, p_hiper) >= self.umbral),
        }

    def explicar_local(self, features: dict[str, float], n_top: int = 3) -> list[dict[str, Any]]:
        """Atribución local por permutación respecto a la mediana del entrenamiento."""
        if not self.artefacto:
            return []
        x = features_vector(features)
        imputer = self.artefacto["imputer"]
        xt = imputer.transform(x.reshape(1, -1))[0]
        mediana = np.array(self.artefacto["referencia"]["media"])
        p_hipo, p_hiper = self._probas(x)

        aportes: list[tuple[str, float]] = []
        for j, nombre in enumerate(FEATURES):
            if nombre in {"hour_sin", "hour_cos", "dia_sin", "dia_cos", "es_sueno"}:
                continue
            xp = xt.copy()
            xp[j] = mediana[j]
            p_hipo_e, p_hiper_e = self._probas(xp)
            delta = abs(p_hipo - p_hipo_e) + abs(p_hiper - p_hiper_e)
            aportes.append((nombre, float(delta)))

        aportes.sort(key=lambda t: t[1], reverse=True)
        resultado = []
        for nombre, delta in aportes[:n_top]:
            valor = float(xt[FEATURES.index(nombre)])
            es_cat = nombre in {"es_diabetico", "familiares_diabetes", "actividad_fisica"}
            detalle = f"Valor: {valor:.2f}" if not es_cat else ""
            resultado.append({
                "senal": FEATURES_LABELS[nombre],
                "aporte": round(delta, 4),
                "detalle": detalle,
            })
        return resultado


def direccion_heuristica(
    frecuencia_cardiaca: float,
    temperatura: float,
    sudoracion_gsr: float | None = None,
    glucosa: float | None = None,
    toma_reciente_medicamento: bool | None = None,
) -> str | None:
    """Fallback sin modelo: patrón hipo/hiper a partir de la lectura puntual y contexto."""
    gsr = float(sudoracion_gsr or 0.0)
    if glucosa is not None:
        if glucosa <= 70.0:
            return _CLASE_HIPO
        if glucosa >= 180.0 and not toma_reciente_medicamento:
            return _CLASE_HIPER

    if frecuencia_cardiaca >= 100 and gsr >= 5.0 and temperatura <= 37.2:
        return _CLASE_HIPO
    if temperatura >= 37.5 and gsr >= 3.0:
        return _CLASE_HIPER
    return None


def construir_resultado_v2(
    paciente_id: str,
    telemetria: dict[str, Any],
    features: dict[str, float] | None,
    modelo: ModeloActivo,
    probabilidad_general: float,
    es_critico_general: bool,
    version: str,
) -> dict[str, Any]:
    """Compone la respuesta v2. Si hay modelo entrenado usa sus probabilidades;
    si no, el fallback heurístico direcciona la probabilidad del baseline."""
    if modelo.disponible and features is not None:
        pred = modelo.predecir(features)
        tipo = pred["tipo"]
        p_hipo = pred["probabilidad_hipo"]
        p_hiper = pred["probabilidad_hiper"]
        probabilidad = pred["probabilidad"]
        es_critico = pred["es_critico"]
        explicacion = modelo.explicar_local(features)
    else:
        direccion = direccion_heuristica(
            telemetria.get("frecuencia_cardiaca", 0),
            telemetria.get("temperatura", 0),
            telemetria.get("sudoracion_gsr") if telemetria.get("sudoracion_gsr") is not None else (telemetria.get("stress_score") or 0) / 5.0,
            telemetria.get("glucosa"),
            telemetria.get("toma_reciente_medicamento"),
        )
        tipo = direccion or "normal"
        if direccion:
            p_hipo = 1.0 if direccion == _CLASE_HIPO else 0.0
            p_hiper = 1.0 if direccion == _CLASE_HIPER else 0.0
            probabilidad = probabilidad_general
            es_critico = es_critico_general
        else:
            p_hipo = 0.0
            p_hiper = 0.0
            probabilidad = probabilidad_general
            es_critico = es_critico_general
        explicacion = []

    horas = 2 if es_critico else (6 if probabilidad >= 0.5 else 24)

    return {
        "paciente_id": paciente_id,
        "riesgo": {
            "tipo": tipo,
            "probabilidad": round(float(probabilidad), 4),
            "probabilidad_hipo": round(float(p_hipo), 4),
            "probabilidad_hiper": round(float(p_hiper), 4),
            "umbral_critico": modelo.umbral,
            "horas_estimadas": horas,
            "es_critico": es_critico,
        },
        "explicacion": explicacion,
        "recomendacion": RECOMENDACIONES.get(tipo, RECOMENDACIONES["normal"]),
        "modelo": {
            "id": modelo.version or "fallback-heuristico",
            "version": version,
            "activo": modelo.disponible,
        },
        "timestamp": datetime.now(timezone.utc),
    }
