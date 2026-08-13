"""Pruebas de estrés y casos de borde para producción (Production Edge Cases & Stress)."""


def test_headers_seguridad_bancaria_presentes(client):
    r = client.get("/health")
    assert r.status_code == 200
    headers = r.headers
    assert "x-correlation-id" in headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert "strict-transport-security" in headers
    assert "content-security-policy" in headers
    assert headers.get("server") == "BioGuard-ML-Core"


def test_prediccion_v2_valores_extremos_criticos(client):
    payload_extremo = {
        "pacienteId": "P-EXTREMO-001",
        "frecuenciaCardiaca": 220,
        "temperatura": 41.5,
        "saturacionOxigeno": 65.0,
        "frecuenciaRespiratoria": 45,
        "presionSistolica": 210,
        "presionDiastolica": 130,
        "glucosa": 450.0,
        "sudoracionGsr": 15.0,
    }
    r = client.post("/api/v2/predicciones", json=payload_extremo)
    assert r.status_code == 201
    data = r.json()
    assert data["riesgo"]["esCritico"] is True
    assert data["riesgo"]["probabilidad"] >= 0.85
    assert data["riesgo"]["tipo"] in {"riesgo_hiper", "riesgo_hipo"}


def test_prediccion_v2_con_actividad_fisica_intensa(client):
    payload_ejercicio = {
        "pacienteId": "P-DEPORTE-001",
        "frecuenciaCardiaca": 130,
        "temperatura": 36.8,
        "saturacionOxigeno": 98.0,
        "frecuenciaRespiratoria": 22,
        "actividadFisica": "intensa",
    }
    r = client.post("/api/v1/predicciones", json=payload_ejercicio)
    assert r.status_code == 201
    data = r.json()
    assert data["esCritico"] is False
    assert "actividad física" in data["explicacion"].lower()


def test_prediccion_v2_con_toma_reciente_medicamento(client):
    payload_med = {
        "pacienteId": "P-MED-001",
        "frecuenciaCardiaca": 80,
        "temperatura": 36.6,
        "saturacionOxigeno": 98.0,
        "frecuenciaRespiratoria": 16,
        "glucosa": 190.0,
        "tomaRecienteMedicamento": True,
    }
    r = client.post("/api/v1/predicciones", json=payload_med)
    assert r.status_code == 201
    data = r.json()
    assert "medicamento" in data["explicacion"].lower()


def test_prediccion_con_senales_opcionales_ausentes(client):
    payload_minimo = {
        "pacienteId": "P-MIN-001",
        "frecuenciaCardiaca": 72,
        "temperatura": 36.5,
        "saturacionOxigeno": 99.0,
        "frecuenciaRespiratoria": 15,
    }
    r = client.post("/api/v2/predicciones", json=payload_minimo)
    assert r.status_code == 201
    assert r.json()["riesgo"]["tipo"] == "normal"


def test_rechazo_presion_arterial_invalida(client):
    payload_invalido = {
        "pacienteId": "P-ERR-001",
        "frecuenciaCardiaca": 80,
        "temperatura": 36.5,
        "saturacionOxigeno": 98.0,
        "frecuenciaRespiratoria": 16,
        "presionSistolica": 80,
        "presionDiastolica": 120,
    }
    r = client.post("/api/v2/predicciones", json=payload_invalido)
    assert r.status_code == 422
