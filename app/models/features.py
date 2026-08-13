"""Ingeniería de features con paridad clínica IRME (RiesgoMetabolicoService.cs)
más estadísticos de ventana fisiológica y contexto circadiano/biométrico."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

# ---- Pesos IRME (paridad con el backend) ----
W_FC_RELATIVA = 0.25
W_HRV_INVERSA = 0.20
W_TEMP_RELATIVA = 0.15
W_REPOSO_POST_EVENTO = 0.15
W_SUENO_RIESGO = 0.10
W_HISTORIAL_PERSONAL = 0.10
W_CONFIRMACION_USUARIO = 0.05

SUENO_INICIO_HORA = 22
SUENO_FIN_HORA = 6
VENTANA_REPOSO_MINUTOS = 60
VENTANA_HISTORIAL_DIAS = 30
MIN_LECTURAS_BASELINE = 50

_ACTIVIDAD_FISICA = {
    "sedentaria": 0.0, "sedentario": 0.0, "ninguna": 0.0,
    "ligera": 1.0, "ligero": 1.0,
    "moderada": 2.0, "moderado": 2.0,
    "activa": 3.0, "activo": 3.0, "intensa": 3.0,
}

FEATURES: list[str] = [
    "fc_media", "fc_std", "fc_min", "fc_max", "fc_slope", "fc_relativa",
    "temp_media", "temp_std", "temp_min", "temp_max", "temp_slope", "temp_relativa",
    "gsr_media", "gsr_max", "gsr_slope", "gsr_altos_ratio",
    "hrv_media", "hrv_inversa", "spo2_min",
    "score_irme",
    "hour_sin", "hour_cos", "dia_sin", "dia_cos", "es_sueno",
    "edad", "imc", "es_diabetico", "familiares_diabetes", "actividad_fisica",
    "minutos_desde_ultima_toma", "eventos_similares_30d", "dias_desde_registro",
]

FEATURES_LABELS: dict[str, str] = {
    "fc_media": "Frecuencia cardiaca (media)",
    "fc_std": "Frecuencia cardiaca (variabilidad)",
    "fc_min": "Frecuencia cardiaca (mínima)",
    "fc_max": "Frecuencia cardiaca (máxima)",
    "fc_slope": "Frecuencia cardiaca (tendencia)",
    "fc_relativa": "Frecuencia cardiaca relativa al baseline",
    "temp_media": "Temperatura (media)",
    "temp_std": "Temperatura (variabilidad)",
    "temp_min": "Temperatura (mínima)",
    "temp_max": "Temperatura (máxima)",
    "temp_slope": "Temperatura (tendencia)",
    "temp_relativa": "Temperatura relativa al baseline",
    "gsr_media": "Sudoración GSR (media)",
    "gsr_max": "Sudoración GSR (máxima)",
    "gsr_slope": "Sudoración GSR (tendencia)",
    "gsr_altos_ratio": "Sudoración GSR (proporción elevada)",
    "hrv_media": "HRV estimado (media)",
    "hrv_inversa": "HRV inversa",
    "spo2_min": "Saturación de oxígeno (mínima)",
    "score_irme": "Score IRME",
    "hour_sin": "Hora del día",
    "hour_cos": "Hora del día",
    "dia_sin": "Día de la semana",
    "dia_cos": "Día de la semana",
    "es_sueno": "Horario de sueño",
    "edad": "Edad",
    "imc": "Índice de masa corporal",
    "es_diabetico": "Diagnóstico de diabetes",
    "familiares_diabetes": "Antecedentes familiares de diabetes",
    "actividad_fisica": "Nivel de actividad física",
    "minutos_desde_ultima_toma": "Minutos desde última toma de medicamento",
    "eventos_similares_30d": "Eventos similares en 30 días",
    "dias_desde_registro": "Días desde registro del paciente",
}

_CIRCULARES = {"hour_sin", "hour_cos", "dia_sin", "dia_cos", "es_sueno"}


def estimar_hrv(pulso_bpm: float, sudoracion_gsr: float) -> float:
    gsr_norm = min(sudoracion_gsr / 10.0, 1.0)
    fc_var = 60000.0 / pulso_bpm if pulso_bpm > 0 else 1000.0
    return fc_var * (1.0 - gsr_norm * 0.3)


def _media(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _gsr_normalizado(rec: dict) -> float:
    """Estrés 0-100 (contrato unificado) mapeado al rango GSR 0-20, con fallback legacy."""
    stress = rec.get("stress_score")
    if stress is not None:
        return float(stress) / 5.0
    return float(rec.get("sudoracion_gsr") or 0.0)


def _slope(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, vals, 1)[0])


def es_horario_sueno(ts: datetime, tz_offset_horas: float = -6.0) -> bool:
    hora_local = (ts.hour + tz_offset_horas) % 24
    return hora_local >= SUENO_INICIO_HORA or hora_local < SUENO_FIN_HORA


def calcular_baseline(
    lecturas: list[dict],
    edad: int | None = None,
    min_lecturas: int = MIN_LECTURAS_BASELINE,
) -> dict[str, float]:
    """Paridad con GetOrCreateBaselineAsync del backend."""
    reposo = [rec for rec in lecturas if (rec.get("pulso_bpm") or 0) <= 100]
    if len(reposo) < min_lecturas:
        fc_est = max(60.0, 220 - (edad or 30) - 20) if edad else 70.0
        return {
            "fc_promedio_reposo": fc_est,
            "hrv_promedio": 50.0,
            "temp_promedio": 36.5,
            "n_lecturas": float(len(reposo)),
        }
    fc = _media([rec["pulso_bpm"] for rec in reposo])
    hrv = _media([estimar_hrv(rec["pulso_bpm"], _gsr_normalizado(rec)) for rec in reposo])
    temp = _media([rec["temperatura_c"] for rec in reposo])
    return {
        "fc_promedio_reposo": fc,
        "hrv_promedio": hrv,
        "temp_promedio": temp,
        "n_lecturas": float(len(reposo)),
    }


def _reposo_post_evento(lecturas: list[dict]) -> float:
    """Paridad con CalcularReposoPostEventoAsync (ventana 60 min, pico >= 0.9*max)."""
    if len(lecturas) < 2:
        return 0.0
    max_fc = max(rec["pulso_bpm"] for rec in lecturas)
    idx = next((idx for idx, rec in enumerate(lecturas) if rec["pulso_bpm"] >= max_fc * 0.9), 0)
    despues = lecturas[idx:]
    antes = lecturas[:idx]
    if len(despues) < 2 or not antes:
        return 0.0
    fc_reposo = _media([rec["pulso_bpm"] for rec in despues])
    fc_base = _media([rec["pulso_bpm"] for rec in antes])
    if fc_base == 0:
        return 0.0
    elevacion = (fc_reposo - fc_base) / fc_base * 100
    return max(0.0, min(elevacion, 100.0))


def _sueno_riesgo(lectura: dict, baseline: dict[str, float]) -> float:
    riesgo = 0.0
    fc_base = baseline["fc_promedio_reposo"]
    if fc_base > 0:
        ratio = lectura["pulso_bpm"] / fc_base
        if ratio > 1.2:
            riesgo += (ratio - 1.2) * 50
    temp_base = baseline["temp_promedio"]
    if temp_base > 0:
        diff = abs(lectura["temperatura_c"] - temp_base)
        if diff > 0.5:
            riesgo += diff * 20
    gsr = _gsr_normalizado(lectura)
    if gsr > 5:
        riesgo += gsr * 5
    return min(riesgo, 100.0)


def _historial_personal(eventos_30d: list[dict], hora_actual: int) -> float:
    if not eventos_30d:
        return 0.0
    similares = sum(
        1
        for e in eventos_30d
        if abs((e.get("fecha_evento") or e.get("timestamp")).hour - hora_actual) <= 2
        and e.get("probabilidad_ml", 0) > 0.7
    )
    return min(similares / len(eventos_30d) * 100, 100.0)


def _ordenar(lecturas: list[dict]) -> list[dict]:
    return sorted(lecturas, key=lambda rec: rec["timestamp"])


def _actividad_fisica_valor(actividad: str | None) -> float:
    if not actividad:
        return 0.0
    return _ACTIVIDAD_FISICA.get(actividad.strip().lower(), 0.0)


def construir_features(
    lecturas: list[dict],
    paciente: dict | None = None,
    medicamento: dict | None = None,
    eventos_30d: list[dict] | None = None,
    tz_offset_horas: float = -6.0,
    baseline_extra: list[dict] | None = None,
) -> dict[str, float]:
    """Calcula el vector de features (orden FEATURES) a partir de la ventana de lecturas."""
    if not lecturas:
        raise ValueError("se requieren lecturas para construir features")
    lecturas = _ordenar(lecturas)

    biometria = (paciente or {}).get("biometria") or {}
    edad = biometria.get("edad")
    baseline = calcular_baseline(baseline_extra if baseline_extra else lecturas, edad=edad)

    fc = [float(rec["pulso_bpm"]) for rec in lecturas]
    temp = [float(rec["temperatura_c"]) for rec in lecturas]
    gsr = [_gsr_normalizado(rec) for rec in lecturas]
    hrv = [
        float(rec.get("hrv"))
        if rec.get("hrv") is not None
        else estimar_hrv(rec["pulso_bpm"], _gsr_normalizado(rec))
        for rec in lecturas
    ]
    spo2 = [float(rec["spo2"]) for rec in lecturas if rec.get("spo2") is not None]

    ultima = lecturas[-1]
    ts = ultima["timestamp"]
    es_sueno = es_horario_sueno(ts, tz_offset_horas)

    fc_base = baseline["fc_promedio_reposo"]
    hrv_base = baseline["hrv_promedio"]
    temp_base = baseline["temp_promedio"]

    fc_relativa = max(0.0, (fc[-1] - fc_base) / fc_base * 100) if fc_base > 0 else 0.0
    hrv_inversa = max(0.0, (hrv_base - hrv[-1]) / hrv_base * 100) if hrv_base > 0 else 0.0
    temp_relativa = abs(temp[-1] - temp_base) / temp_base * 100 if temp_base > 0 else 0.0
    reposo_post_evento = _reposo_post_evento(lecturas)
    sueno_riesgo = _sueno_riesgo(ultima, baseline) if es_sueno else 0.0
    historial = _historial_personal(eventos_30d or [], ts.hour)

    score_irme = (
        W_FC_RELATIVA * min(fc_relativa, 100)
        + W_HRV_INVERSA * min(hrv_inversa, 100)
        + W_TEMP_RELATIVA * min(temp_relativa, 100)
        + W_REPOSO_POST_EVENTO * reposo_post_evento
        + W_SUENO_RIESGO * sueno_riesgo
        + W_HISTORIAL_PERSONAL * historial
        + W_CONFIRMACION_USUARIO * 0.0
    )
    score_irme = max(0.0, min(score_irme, 100.0))

    estatura_m = (biometria.get("estatura_cm") or 0) / 100.0
    imc = (biometria.get("peso_kg") or 0) / (estatura_m ** 2) if estatura_m > 0 else 0.0

    minutos_toma = -1.0
    if medicamento and medicamento.get("ultima_toma"):
        minutos_toma = max(0.0, (ts - medicamento["ultima_toma"]).total_seconds() / 60.0)

    dias_registro = -1.0
    if paciente and paciente.get("fecha_registro"):
        dias_registro = max(0.0, (ts - paciente["fecha_registro"]).days)

    hour = float(ts.hour) + float(ts.minute) / 60.0
    dia = float(ts.weekday())

    return {
        "fc_media": _media(fc),
        "fc_std": float(np.std(fc)),
        "fc_min": min(fc),
        "fc_max": max(fc),
        "fc_slope": _slope(fc),
        "fc_relativa": fc_relativa,
        "temp_media": _media(temp),
        "temp_std": float(np.std(temp)),
        "temp_min": min(temp),
        "temp_max": max(temp),
        "temp_slope": _slope(temp),
        "temp_relativa": temp_relativa,
        "gsr_media": _media(gsr),
        "gsr_max": max(gsr),
        "gsr_slope": _slope(gsr),
        "gsr_altos_ratio": sum(1 for g in gsr if g > 5.0) / len(gsr),
        "hrv_media": _media(hrv),
        "hrv_inversa": hrv_inversa,
        "spo2_min": min(spo2) if spo2 else float("nan"),
        "score_irme": score_irme,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dia_sin": float(np.sin(2 * np.pi * dia / 7.0)),
        "dia_cos": float(np.cos(2 * np.pi * dia / 7.0)),
        "es_sueno": 1.0 if es_sueno else 0.0,
        "edad": float(edad or 0.0),
        "imc": imc,
        "es_diabetico": 1.0 if biometria.get("es_diabetico") else 0.0,
        "familiares_diabetes": 1.0 if biometria.get("familiares_diabetes") else 0.0,
        "actividad_fisica": _actividad_fisica_valor(biometria.get("actividad_fisica")),
        "minutos_desde_ultima_toma": minutos_toma,
        "eventos_similares_30d": historial / 100.0,
        "dias_desde_registro": dias_registro,
    }


def features_vector(features: dict[str, float]) -> np.ndarray:
    """Vector numpy en el orden canónico FEATURES (NaN si falta)."""
    return np.array([features.get(f, float("nan")) for f in FEATURES], dtype=float)


def normalizar_ts(doc: dict) -> dict:
    """Normaliza documentos Mongo a dict plano con timestamps aware en UTC."""
    ts = doc.get("timestamp")
    if ts is not None and not hasattr(ts, "tzinfo"):
        ts = ts.replace(tzinfo=timezone.utc)
    out = dict(doc)
    out["timestamp"] = ts
    return out
