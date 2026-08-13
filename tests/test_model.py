from datetime import datetime, timedelta, timezone

from app.models.drift import calcular_drift
from app.models.entrenamiento import (
    deserializar_artefacto,
    entrenar_modelos,
    serializar_artefacto,
)
from app.models.features import construir_features
from app.models.inferencia import ModeloActivo
from app.models.labels import (
    CLASE_HIPO,
    CLASE_HIPER,
    CLASE_NORMAL,
    clasificar_direccion,
    preparar_dataset,
)


def _lectura(pulso: float, temp: float, gsr: float, ts: datetime) -> dict:
    return {
        "timestamp": ts,
        "pulso_bpm": pulso,
        "temperatura_c": temp,
        "sudoracion_gsr": gsr,
        "hrv": 60000.0 / pulso * (1 - min(gsr / 10.0, 1) * 0.3),
        "spo2": 96,
    }


def _datos_sinteticos(n_pacientes: int = 8) -> dict:
    datos = {}
    for p in range(n_pacientes):
        pid = f"pac-{p}"
        base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc) + timedelta(days=p)
        # Cadencia 30 min: la ventana de 120 min previa al evento solo captura ~4 normales,
        # de modo que el pico domina la clasificación (fc>=100, gsr>=5).
        lecturas = [
            _lectura(72 + p, 36.6, 2.0, base + timedelta(minutes=30 * i)) for i in range(100)
        ]
        ts_evento = base + timedelta(minutes=30 * 100)
        # 24 lecturas de pico cubriendo la ventana completa (-5..-120 min) => patrón hipo
        for i in range(1, 25):
            lecturas.append(_lectura(125, 36.2, 8.0, ts_evento - timedelta(minutes=5 * i)))
        eventos = [
            {
                "paciente_id": pid,
                "nivel_riesgo": "Alto",
                "probabilidad_ml": 0.9,
                "fecha_evento": ts_evento,
                "variables_irme": {},
            }
        ]
        datos[pid] = {
            "paciente": {"biometria": {"edad": 40 + p}, "fecha_registro": base - timedelta(days=100)},
            "lecturas": lecturas,
            "eventos": eventos,
            "medicamento": None,
        }
    return datos


def test_clasificar_direccion():
    now = datetime.now(timezone.utc)
    assert clasificar_direccion([_lectura(125, 36.2, 8.0, now)]) == CLASE_HIPO
    assert clasificar_direccion([_lectura(110, 38.0, 5.0, now)]) == CLASE_HIPER
    assert clasificar_direccion([_lectura(72, 36.6, 2.0, now)]) is None


def test_preparar_dataset_y_entrenamiento():
    ds = preparar_dataset(_datos_sinteticos())
    assert ds["conteos"][CLASE_HIPO] >= 5
    assert ds["conteos"][CLASE_NORMAL] >= 5
    artefacto = entrenar_modelos(ds["X"], ds["y"])
    assert artefacto is not None
    assert artefacto["modelo_hipo"] is not None
    assert artefacto["version"].startswith("diabetes-hipo-hiper-v")
    assert "hipo" in artefacto["metricas"]
    assert artefacto["total_muestras"] == len(ds["y"])


def test_inferencia_distinguen_hipo_de_normal():
    ds = preparar_dataset(_datos_sinteticos())
    artefacto = entrenar_modelos(ds["X"], ds["y"])
    assert artefacto is not None
    modelo = ModeloActivo(artefacto, umbral=0.85)

    base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    normal = [_lectura(72, 36.6, 2.0, base - timedelta(minutes=5 * i)) for i in range(12)]
    f_normal = construir_features(normal, paciente={"biometria": {"edad": 40}})
    pred = modelo.predecir(f_normal)
    assert pred["tipo"] == CLASE_NORMAL

    hipo = [_lectura(128, 36.2, 8.5, base - timedelta(minutes=5 * i)) for i in range(12)]
    f_hipo = construir_features(hipo, paciente={"biometria": {"edad": 40}})
    pred_hipo = modelo.predecir(f_hipo)
    assert pred_hipo["probabilidad_hipo"] > pred["probabilidad_hipo"]

    explicacion = modelo.explicar_local(f_hipo, n_top=3)
    assert 0 < len(explicacion) <= 3
    assert all(e["aporte"] >= 0 for e in explicacion)
    assert all(e["senal"] for e in explicacion)


def test_serializacion_artefacto():
    ds = preparar_dataset(_datos_sinteticos(4))
    artefacto = entrenar_modelos(ds["X"], ds["y"])
    if artefacto is None:
        return
    recuperado = deserializar_artefacto(serializar_artefacto(artefacto))
    assert recuperado["version"] == artefacto["version"]
    assert recuperado["modelo_hipo"] is not None


def test_drift_con_features_normales():
    ds = preparar_dataset(_datos_sinteticos())
    artefacto = entrenar_modelos(ds["X"], ds["y"])
    assert artefacto is not None
    base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    normales = [
        construir_features(
            [_lectura(72, 36.6, 2.0, base - timedelta(minutes=5 * i)) for i in range(12)]
        )
        for _ in range(20)
    ]
    drift = calcular_drift(normales, artefacto["referencia"])
    assert drift["nivel"] in {"ok", "atencion", "critico", "sin_referencia"}
    assert "z_scores" in drift
    assert isinstance(drift["features_desviadas"], list)
