from app.schemas.telemetria import TelemetriaEntrada

URL = "/api/v1/predicciones"

PAYLOAD_VALIDO = {
    "pacienteId": "P-001",
    "frecuenciaCardiaca": 95,
    "temperatura": 37.2,
    "saturacionOxigeno": 96.0,
    "frecuenciaRespiratoria": 18,
    "presionSistolica": 120,
    "presionDiastolica": 80,
    "glucosa": 105.0,
    "dispositivo": "smartwatch-v1",
}

PAYLOAD_CRITICO = {
    "pacienteId": "P-999",
    "frecuenciaCardiaca": 170,
    "temperatura": 41.0,
    "saturacionOxigeno": 70,
    "frecuenciaRespiratoria": 40,
    "presionSistolica": 190,
    "presionDiastolica": 110,
    "glucosa": 350,
    "dispositivo": "smartwatch-v1",
}

PAYLOAD_SANO = {
    **PAYLOAD_VALIDO,
    "frecuenciaCardiaca": 75,
    "temperatura": 36.8,
    "saturacionOxigeno": 98,
    "frecuenciaRespiratoria": 16,
}


def test_v1_senal_saludable_probabilidad_baja(client):
    r = client.post(URL, json=PAYLOAD_SANO)
    assert r.status_code == 201
    data = r.json()
    assert data["probabilidad"] == 0.0474
    assert data["esCritico"] is False
    assert data["nivelRiesgo"] == "BAJO"
    assert data["umbralCritico"] == 0.85
    assert data["modeloId"] == "baseline-v0"
    assert data["contribuciones"]
    assert data["explicacion"]


def test_v1_senal_critica_probabilidad_alta(client):
    r = client.post(URL, json=PAYLOAD_CRITICO)
    assert r.status_code == 201
    data = r.json()
    assert data["probabilidad"] == 0.9324
    assert data["esCritico"] is True
    assert data["nivelRiesgo"] == "CRITICO"


def test_v1_senal_critica_supera_a_sana(client):
    sano = client.post(URL, json=PAYLOAD_SANO).json()
    critico = client.post(URL, json=PAYLOAD_CRITICO).json()
    assert sano["probabilidad"] < critico["probabilidad"]
    assert sano["esCritico"] is not critico["esCritico"]


def test_v1_acepta_snake_case(client):
    payload = {**PAYLOAD_SANO}
    payload["paciente_id"] = payload.pop("pacienteId")
    payload["frecuencia_cardiaca"] = payload.pop("frecuenciaCardiaca")
    payload["saturacion_oxigeno"] = payload.pop("saturacionOxigeno")
    payload["frecuencia_respiratoria"] = payload.pop("frecuenciaRespiratoria")
    r = client.post(URL, json=payload)
    assert r.status_code == 201
    assert r.json()["pacienteId"] == "P-001"


def test_v1_sin_señales_opcionales(client):
    payload = {
        "paciente_id": "P-003",
        "frecuencia_cardiaca": 82,
        "temperatura": 36.9,
        "saturacion_oxigeno": 97.0,
        "frecuencia_respiratoria": 17,
    }
    r = client.post(URL, json=payload)
    assert r.status_code == 201
    assert 0.0 <= r.json()["probabilidad"] <= 1.0


def test_v1_422_falta_paciente(client):
    payload = {**PAYLOAD_SANO}
    del payload["pacienteId"]
    assert client.post(URL, json=payload).status_code == 422


def test_v1_422_valor_fuera_de_rango(client):
    assert client.post(URL, json={**PAYLOAD_SANO, "frecuenciaCardiaca": 1000}).status_code == 422


def test_v1_422_campos_extra(client):
    assert client.post(URL, json={**PAYLOAD_SANO, "campoDesconocido": "x"}).status_code == 422


def test_v1_422_presion_inconsistente(client):
    payload = {**PAYLOAD_SANO, "presionSistolica": 80, "presionDiastolica": 120}
    assert client.post(URL, json=payload).status_code == 422


def test_v1_422_payload_vacio(client):
    assert client.post(URL, json={}).status_code == 422


def test_telemetria_modelo_valida_alias_camel(client):
    t = TelemetriaEntrada(
        pacienteId="P-001",
        frecuenciaCardiaca=75,
        temperatura=36.8,
        saturacionOxigeno=98,
        frecuenciaRespiratoria=16,
    )
    assert t.paciente_id == "P-001"
    assert t.frecuencia_cardiaca == 75
