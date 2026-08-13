"""Etiquetas clínicas: normal / riesgo_hipo / riesgo_hiper.

La dirección (hipo vs hiper) se deriva de patrones fisiológicos porque el sistema
es de proxies (pulso, temperatura, GSR) sin glucómetro directo. Los eventos
confirmados por usuario/cuidador tienen prioridad como "golden dataset".
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

from app.models.features import construir_features

CLASE_NORMAL = "normal"
CLASE_HIPO = "riesgo_hipo"
CLASE_HIPER = "riesgo_hiper"
CLASES = [CLASE_NORMAL, CLASE_HIPO, CLASE_HIPER]

VENTANA_EVENTO_MIN = 120
VENTANA_EVENTO_MINUS = timedelta(minutes=VENTANA_EVENTO_MIN)
VENTANA_ESPERA_NEGATIVOS_MIN = 120


def evento_confirmado(evento: dict) -> bool:
    """Evento validado por humano (atendido o con confirmación registrada)."""
    if evento.get("atendida"):
        return True
    if evento.get("acciones_tomadas"):
        return True
    vars_irme = evento.get("variables_irme") or {}
    return float(vars_irme.get("confirmacion_usuario") or 0) > 0


def _gsr_normalizado(rec: dict) -> float:
    """Estrés 0-100 (contrato unificado) mapeado al rango GSR 0-20, con fallback legacy."""
    stress = rec.get("stress_score")
    if stress is not None:
        return float(stress) / 5.0
    return float(rec.get("sudoracion_gsr") or 0.0)


def clasificar_direccion(
    lecturas_ventana: list[dict],
    nivel_riesgo: str | None = None,
) -> str | None:
    """Heurística clínica de dirección del evento metabólico.

    - Hipo: taquicardia + sudoración elevada + temperatura normal/baja.
    - Hiper: temperatura elevada sostenida (+ sudoración).
    """
    if not lecturas_ventana:
        return None
    fc = sum(rec["pulso_bpm"] for rec in lecturas_ventana) / len(lecturas_ventana)
    temp = sum(rec["temperatura_c"] for rec in lecturas_ventana) / len(lecturas_ventana)
    gsr = sum(_gsr_normalizado(rec) for rec in lecturas_ventana) / len(lecturas_ventana)

    if fc >= 100 and gsr >= 5 and temp <= 37.2:
        return CLASE_HIPO
    if temp >= 37.5 and gsr >= 3:
        return CLASE_HIPER
    if nivel_riesgo and nivel_riesgo.lower() in {"critico", "crítico", "alto"}:
        if temp >= 37.5:
            return CLASE_HIPER
        if fc >= 100 and gsr >= 5:
            return CLASE_HIPO
    return None


def preparar_dataset(
    datos_pacientes: dict[str, dict],
    seed: int = 42,
    negativo_ratio: float = 2.0,
    min_lecturas_ventana: int = 3,
) -> dict[str, Any]:
    """Convierte datos crudos (pacientes/lecturas/eventos) en muestras de entrenamiento.

    Returns:
        {
          "X": list[dict] (features en FEATURES order),
          "y": list[str],
          "confirmados": list[bool],
          "conteos": dict,
        }
    """
    rng = random.Random(seed)
    muestras: list[tuple[dict, str, bool]] = []

    for datos in datos_pacientes.values():
        paciente = datos.get("paciente")
        medicamento = datos.get("medicamento")
        lecturas = sorted(datos.get("lecturas", []), key=lambda item: item["timestamp"])
        eventos = datos.get("eventos", [])
        if len(lecturas) < min_lecturas_ventana:
            continue

        tiempos_evento = [e["fecha_evento"] for e in eventos]
        positivos_por_clase = {CLASE_HIPO: 0, CLASE_HIPER: 0}

        for evento in eventos:
            ts_evento = evento["fecha_evento"]
            ventana = [
                rec for rec in lecturas if ts_evento - VENTANA_EVENTO_MINUS <= rec["timestamp"] < ts_evento
            ]
            if len(ventana) < min_lecturas_ventana:
                continue
            clase = clasificar_direccion(ventana, evento.get("nivel_riesgo"))
            if clase not in (CLASE_HIPO, CLASE_HIPER):
                continue
            try:
                feats = construir_features(
                    ventana,
                    paciente=paciente,
                    medicamento=medicamento,
                    eventos_30d=[e for e in eventos if e["fecha_evento"] >= ts_evento - timedelta(days=30)],
                )
            except ValueError:
                continue
            muestras.append((feats, clase, evento_confirmado(evento)))
            positivos_por_clase[clase] += 1

        total_positivos = sum(positivos_por_clase.values())
        if total_positivos == 0:
            continue
        objetivo_negativos = int(total_positivos * negativo_ratio)

        def _cerca_de_evento(ts: Any) -> bool:
            limit_sec = VENTANA_ESPERA_NEGATIVOS_MIN * 60
            return any(abs((ts - t).total_seconds()) < limit_sec for t in tiempos_evento)

        negativos = 0
        intentos = 0
        while negativos < objetivo_negativos and intentos < 200:
            intentos += 1
            i = rng.randrange(len(lecturas))
            inicio = lecturas[i]["timestamp"]
            desde = inicio - timedelta(minutes=90)
            ventana = [rec for rec in lecturas if desde <= rec["timestamp"] <= inicio]
            if len(ventana) < min_lecturas_ventana:
                continue
            try:
                feats = construir_features(
                    ventana,
                    paciente=paciente,
                    medicamento=medicamento,
                    eventos_30d=[e for e in eventos if e["fecha_evento"] >= inicio - timedelta(days=30)],
                )
            except ValueError:
                continue
            muestras.append((feats, CLASE_NORMAL, True))
            negativos += 1

    if not muestras:
        return {"X": [], "y": [], "confirmados": [], "conteos": {}}

    X, y, confirmados = zip(*muestras)
    conteos = {c: sum(1 for v in y if v == c) for c in CLASES}
    return {
        "X": list(X),
        "y": list(y),
        "confirmados": list(confirmados),
        "conteos": conteos,
    }
