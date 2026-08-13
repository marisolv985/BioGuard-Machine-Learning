def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["estado"] == "ok"
    assert data["servicio"] == "BioGuard ML Service"
    assert data["umbralCritico"] == 0.85
    assert data["modeloActivo"]
    assert data["enviroment"]


def test_ready_sin_mongo(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["estado"] == "listo"
