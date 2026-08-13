from datetime import datetime, timedelta, timezone

from app.models.features import (
    FEATURES,
    calcular_baseline,
    construir_features,
    es_horario_sueno,
    features_vector,
)


def _lectura(pulso: float, temp: float, gsr: float = 2.0, ts: datetime | None = None) -> dict:
    return {
        "timestamp": ts or datetime.now(timezone.utc),
        "pulso_bpm": pulso,
        "temperatura_c": temp,
        "sudoracion_gsr": gsr,
        "hrv": 60000.0 / pulso * (1 - min(gsr / 10.0, 1) * 0.3),
        "spo2": 97,
    }


def _serie_normal(horas: int = 10) -> list[dict]:
    ahora = datetime.now(timezone.utc)
    return [
        _lectura(72, 36.6, 2.0, ahora - timedelta(minutes=5 * (horas * 12 - i)))
        for i in range(horas * 12)
    ]


def test_baseline_poblacional_sin_historial():
    lecturas = _serie_normal(1)  # 12 lecturas < 50 => estimación por edad
    baseline = calcular_baseline(lecturas, edad=50)
    assert baseline["fc_promedio_reposo"] == 150  # 220 - 50 - 20
    assert baseline["n_lecturas"] == 12


def test_baseline_personal_con_historial():
    baseline = calcular_baseline(_serie_normal(10))
    assert baseline["n_lecturas"] == 120
    assert 65 <= baseline["fc_promedio_reposo"] <= 80


def test_features_normal_bajas():
    f = construir_features(_serie_normal(10), paciente={"biometria": {"edad": 40}})
    assert f["fc_relativa"] < 10
    assert f["score_irme"] < 20
    assert f["temp_media"] == 36.6
    assert f["es_diabetico"] == 0.0


def test_features_reflejan_alteracion_en_ventana():
    ts = datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc)  # horario de sueño (-6h => 23h)
    ventana = [_lectura(128, 36.3, 8.5, ts - timedelta(minutes=5 * i)) for i in range(12)]
    f = construir_features(ventana, paciente={"biometria": {"edad": 40}}, baseline_extra=_serie_normal(10))
    assert f["fc_media"] > 90
    assert f["gsr_max"] >= 8.5
    assert f["score_irme"] > 20


def test_features_vector_orden_canonico():
    f = construir_features(_serie_normal(10))
    v = features_vector(f)
    assert len(v) == len(FEATURES)
    assert v[FEATURES.index("fc_media")] == f["fc_media"]
    assert v[FEATURES.index("temp_media")] == f["temp_media"]


def test_es_horario_sueno():
    assert es_horario_sueno(datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc)) is True
    assert es_horario_sueno(datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)) is False


def test_imc_y_diabetes_se_calculan():
    paciente = {"biometria": {"edad": 45, "peso_kg": 80, "estatura_cm": 175, "es_diabetico": True}}
    f = construir_features(_serie_normal(10), paciente=paciente)
    assert abs(f["imc"] - 26.12) < 0.1
    assert f["es_diabetico"] == 1.0
    assert f["edad"] == 45
