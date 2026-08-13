from datetime import datetime, timedelta, timezone

from tests.conftest import sembrar_paciente


def _riesgo(tipo: str, proba: float) -> dict:
    return {
        "tipo": tipo,
        "probabilidad": proba,
        "probabilidad_hipo": proba if tipo == "riesgo_hipo" else 0.0,
        "probabilidad_hiper": proba if tipo == "riesgo_hiper" else 0.0,
        "umbral_critico": 0.85,
        "horas_estimadas": 2,
        "es_critico": proba >= 0.85,
    }


def _sembrar_predicciones(fake_mongo):
    ahora = datetime.now(timezone.utc)
    # Horas [1, 2, 26]: dos dentro del último día (buckets horarios) y una en el
    # día anterior (bucket diario distinto) para los rangos dia/semana.
    for i, (tipo, proba) in enumerate(
        [("normal", 0.05), ("riesgo_hipo", 0.71), ("riesgo_hiper", 0.6)]
    ):
        fake_mongo.predicciones.docs.append(
            {
                "_id": f"pred-{i}",
                "paciente_id": "P-01",
                "riesgo": _riesgo(tipo, proba),
                "explicacion": [{"senal": "Frecuencia cardiaca", "aporte": 0.3, "detalle": ""}],
                "recomendacion": "ok",
                "modelo": {"id": "test", "version": "2.0.0", "activo": False},
                "timestamp": ahora - timedelta(hours=[1, 2, 26][i]),
            }
        )


def test_riesgo_actual(client_con_db, fake_mongo):
    _sembrar_predicciones(fake_mongo)
    r = client_con_db.get("/api/v2/pacientes/P-01/riesgo-actual")
    assert r.status_code == 200
    data = r.json()
    assert data["tipo"] == "normal"
    assert data["probabilidad"] == 0.05


def test_riesgo_actual_sin_datos(client_con_db):
    r = client_con_db.get("/api/v2/pacientes/inexistente/riesgo-actual")
    assert r.status_code == 200
    assert r.json() is None


def test_tendencias(client_con_db, fake_mongo):
    _sembrar_predicciones(fake_mongo)
    r = client_con_db.get("/api/v2/pacientes/P-01/tendencias?rango=dia")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 2
    for punto in data:
        assert 0.0 <= punto["riesgoPromedio"] <= 1.0
        assert punto["nPredicciones"] >= 1


def test_tendencias_semana(client_con_db, fake_mongo):
    _sembrar_predicciones(fake_mongo)
    r = client_con_db.get("/api/v2/pacientes/P-01/tendencias?rango=semana")
    assert r.status_code == 200
    assert len(r.json()) >= 2


def test_historial_paginado(client_con_db, fake_mongo):
    _sembrar_predicciones(fake_mongo)
    r = client_con_db.get("/api/v2/pacientes/P-01/historial?pagina=1&paginaTamano=2")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    assert len(data["items"]) == 2
    assert data["items"][0]["prediccionId"] == "pred-0"
    assert data["items"][0]["tipo"] == "normal"


def test_resumen(client_con_db, fake_mongo):
    _sembrar_predicciones(fake_mongo)
    r = client_con_db.get("/api/v2/pacientes/P-01/resumen")
    assert r.status_code == 200
    data = r.json()
    assert data["pacienteId"] == "P-01"
    assert data["riesgoActual"]["tipo"] == "normal"
    assert data["tendencia"]
    assert data["explicacion"][0]["senal"] == "Frecuencia cardiaca"


def test_explicabilidad(client_con_db, fake_mongo):
    _sembrar_predicciones(fake_mongo)
    r = client_con_db.get("/api/v2/pacientes/P-01/explicabilidad/pred-1")
    assert r.status_code == 200
    data = r.json()
    assert data and data[0]["senal"] == "Frecuencia cardiaca"


def test_explicabilidad_sin_datos(client_con_db):
    r = client_con_db.get("/api/v2/pacientes/P-01/explicabilidad/inexistente")
    assert r.status_code == 200
    assert r.json() == []


def test_confirmar_evento(client_con_db, fake_mongo):
    r = client_con_db.post(
        "/api/internal/eventos/confirmar",
        json={"pacienteId": "P-01", "eventoId": "ev-1", "confirmado": True, "nota": "síntomas reales"},
    )
    assert r.status_code == 204
    docs = fake_mongo.eventos_confirmados.docs
    assert len(docs) == 1
    assert docs[0]["confirmado"] is True
    assert docs[0]["paciente_id"] == "P-01"


def test_purga_derecho_al_olvido(client_con_db, fake_mongo):
    _sembrar_predicciones(fake_mongo)
    fake_mongo.features.docs.append({"paciente_id": "P-01", "features": {"fc_media": 80}})
    r = client_con_db.delete("/api/internal/pacientes/P-01")
    assert r.status_code == 204
    assert fake_mongo.predicciones.docs == []
    assert fake_mongo.features.docs == []


def _sembrar_para_entrenar(fake_mongo, n_pacientes: int = 20):
    ahora = datetime.now(timezone.utc)
    for p in range(n_pacientes):
        pid = f"pac-{p}"
        # Cadencia 30 min: la ventana de 120 min previa al evento solo captura ~4 normales,
        # de modo que el pico domina (fc>=100, gsr>=5) y se etiqueta como hipo.
        base = sembrar_paciente(
            fake_mongo,
            pid,
            n_lecturas=100,
            base=ahora - timedelta(days=30) + timedelta(days=p),
            cadencia_min=30,
        )
        ts_evento = base + timedelta(minutes=30 * 100)
        for i in range(1, 25):
            fake_mongo.lecturas.docs.append(
                {
                    "meta": {"paciente_id": pid},
                    "timestamp": ts_evento - timedelta(minutes=5 * i),
                    "pulso_bpm": 125,
                    "temperatura_c": 36.2,
                    "sudoracion_gsr": 8.0,
                    "hrv": 40.0,
                    "spo2": 96,
                }
            )
        fake_mongo.eventos.docs.append(
            {
                "paciente_id": pid,
                "nivel_riesgo": "Alto",
                "probabilidad_ml": 0.9,
                "fecha_evento": ts_evento,
                "variables_irme": {},
            }
        )


def test_entrenar_via_api(client_con_db, fake_mongo):
    _sembrar_para_entrenar(fake_mongo)
    r = client_con_db.post("/api/v2/modelos/entrenar", json={"descripcion": "smoke", "retenerActivo": False})
    assert r.status_code == 200
    data = r.json()
    assert data["version"].startswith("diabetes-hipo-hiper-v")
    assert data["activo"] is True
    assert data["totalMuestras"] > 0
    assert len(fake_mongo.modelos.docs) == 1

    r2 = client_con_db.get("/api/v2/modelos/activo")
    assert r2.status_code == 200
    activo = r2.json()
    assert activo["version"] == data["version"]
    assert activo["activo"] is True
    assert activo["features"]


def test_listar_modelos_vacio(client_con_db):
    r = client_con_db.get("/api/v2/modelos")
    assert r.status_code == 200
    assert r.json() == []


def test_modelo_activo_fallback(client_con_db):
    r = client_con_db.get("/api/v2/modelos/activo")
    assert r.status_code == 200
    data = r.json()
    assert data["activo"] is False
    assert data["tipo"] == "fallback-heuristico"
