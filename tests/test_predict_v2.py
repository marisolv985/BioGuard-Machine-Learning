from tests.conftest import sembrar_paciente

URL = "/api/v2/predicciones"

PAYLOAD_HIPER = {
    "pacienteId": "P-001",
    "frecuenciaCardiaca": 118,
    "temperatura": 38.6,
    "saturacionOxigeno": 91,
    "frecuenciaRespiratoria": 24,
    "presionSistolica": 165,
    "presionDiastolica": 102,
    "sudoracionGsr": 6.5,
    "hrv": 45.0,
}

PAYLOAD_HIPO = {
    "pacienteId": "P-002",
    "frecuenciaCardiaca": 125,
    "temperatura": 36.4,
    "saturacionOxigeno": 96,
    "frecuenciaRespiratoria": 22,
    "sudoracionGsr": 8.0,
    "hrv": 38.0,
}

PAYLOAD_SANO = {
    "pacienteId": "P-003",
    "frecuenciaCardiaca": 75,
    "temperatura": 36.8,
    "saturacionOxigeno": 98,
    "frecuenciaRespiratoria": 16,
    "sudoracionGsr": 2.0,
    "hrv": 60.0,
}


def test_v2_respuesta_completa(client):
    r = client.post(URL, json=PAYLOAD_HIPER)
    assert r.status_code == 201
    data = r.json()
    assert data["pacienteId"] == "P-001"
    assert "riesgo" in data
    assert data["riesgo"]["tipo"] in {"normal", "riesgo_hipo", "riesgo_hiper"}
    assert 0.0 <= data["riesgo"]["probabilidad"] <= 1.0
    assert "probabilidadHipo" in data["riesgo"]
    assert "probabilidadHiper" in data["riesgo"]
    assert "horasEstimadas" in data["riesgo"]
    assert data["recomendacion"]
    assert data["modelo"]["id"]
    assert data["timestamp"]


def test_v2_fallback_detecta_hiper(client):
    data = client.post(URL, json=PAYLOAD_HIPER).json()
    assert data["riesgo"]["tipo"] == "riesgo_hiper"


def test_v2_fallback_detecta_hipo(client):
    data = client.post(URL, json=PAYLOAD_HIPO).json()
    assert data["riesgo"]["tipo"] == "riesgo_hipo"


def test_v2_fallback_normal(client):
    data = client.post(URL, json=PAYLOAD_SANO).json()
    assert data["riesgo"]["tipo"] == "normal"


def test_v2_usa_features_de_bd_si_esta_configurado(client_con_db, fake_mongo):
    sembrar_paciente(fake_mongo, "P-003", n_lecturas=60)
    r = client_con_db.post(URL, json=PAYLOAD_SANO)
    assert r.status_code == 201
    assert r.json()["riesgo"]["tipo"] == "normal"


def test_v2_persiste_prediccion(client, fake_mongo):
    r = client.post(URL, json=PAYLOAD_HIPER)
    assert r.status_code == 201
    docs = fake_mongo.predicciones.docs
    assert len(docs) == 1
    assert docs[0]["paciente_id"] == "P-001"
    assert docs[0]["riesgo"]["tipo"] == "riesgo_hiper"
    assert "timestamp" in docs[0]


def test_v2_rechaza_payload_invalido(client):
    payload = {**PAYLOAD_HIPER, "frecuenciaCardiaca": 1000}
    assert client.post(URL, json=payload).status_code == 422
