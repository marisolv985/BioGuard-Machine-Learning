from datetime import datetime, timedelta, timezone

import jwt
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import crear_token_servicio
from app.main import build_app

SECRETO = "clave-servicio-muy-segura-min-32-caracteres-1234"

PAYLOAD = {
    "pacienteId": "P-999",
    "frecuenciaCardiaca": 75,
    "temperatura": 36.8,
    "saturacionOxigeno": 98,
    "frecuenciaRespiratoria": 16,
}


def _app_con_seguridad(fake_mongo):
    s = Settings(service_token_secret=SECRETO)
    return TestClient(build_app(mongo=fake_mongo, settings_override=s))


def _token(payload_extra: dict | None = None) -> str:
    ahora = datetime.now(timezone.utc)
    payload = {
        "iss": "BioGuardApi",
        "aud": "ml-service",
        "tipo": "servicio",
        "iat": int(ahora.timestamp()),
        "exp": int(ahora.timestamp()) + 120,
        **(payload_extra or {}),
    }
    return jwt.encode(payload, SECRETO, algorithm="HS256")


def test_sin_token_401(fake_mongo):
    c = _app_con_seguridad(fake_mongo)
    assert c.post("/api/v1/predicciones", json=PAYLOAD).status_code == 401


def test_token_invalido_401(fake_mongo):
    c = _app_con_seguridad(fake_mongo)
    r = c.post("/api/v1/predicciones", json=PAYLOAD, headers={"Authorization": "Bearer token-malo"})
    assert r.status_code == 401


def test_token_valido_201(fake_mongo):
    c = _app_con_seguridad(fake_mongo)
    s = Settings(service_token_secret=SECRETO)
    token = crear_token_servicio(s)
    r = c.post("/api/v1/predicciones", json=PAYLOAD, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 201


def test_audiencia_incorrecta_401(fake_mongo):
    c = _app_con_seguridad(fake_mongo)
    token = _token({"aud": "otra-audiencia"})
    r = c.post("/api/v1/predicciones", json=PAYLOAD, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_tipo_usuario_403(fake_mongo):
    c = _app_con_seguridad(fake_mongo)
    token = _token({"tipo": "usuario"})
    r = c.post("/api/v1/predicciones", json=PAYLOAD, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403


def test_expira_401(fake_mongo):
    c = _app_con_seguridad(fake_mongo)
    token = _token({"exp": int((datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp())})
    r = c.post("/api/v1/predicciones", json=PAYLOAD, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_sin_secreto_no_exige_token(client):
    assert client.post("/api/v1/predicciones", json=PAYLOAD).status_code == 201
