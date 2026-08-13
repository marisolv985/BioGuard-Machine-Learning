"""Fake en memoria del subset de motor/pymongo que usa el servicio + fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import build_app


def _get_path(d: dict, path: str) -> Any:
    actual: Any = d
    for parte in path.split("."):
        if not isinstance(actual, dict) or parte not in actual:
            return None
        actual = actual[parte]
    return actual


def _match(d: dict, filtro: dict) -> bool:
    for k, v in (filtro or {}).items():
        if k == "_id":
            actual = d.get("_id")
            if isinstance(v, ObjectId):
                if actual != v:
                    return False
            elif str(actual) != str(v):
                return False
            continue
        actual = _get_path(d, k)
        if isinstance(v, dict):
            if "$gte" in v and (actual is None or actual < v["$gte"]):
                return False
            if "$lte" in v and (actual is None or actual > v["$lte"]):
                return False
            if "$in" in v and actual not in v["$in"]:
                return False
            if "$ne" in v and actual == v["$ne"]:
                return False
            continue
        if actual != v:
            return False
    return True


def _set_path(d: dict, path: str, valor: Any) -> None:
    partes = path.split(".")
    cur = d
    for parte in partes[:-1]:
        cur = cur.setdefault(parte, {})
    cur[partes[-1]] = valor


class FakeCursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = list(docs)

    def sort(self, key: str, direction: int = 1) -> "FakeCursor":
        if isinstance(key, (list, tuple)):
            keys: list[tuple[str, int]] = [(k, dir_) for k, dir_ in key]
        else:
            keys = [(key, direction)]

        def _clave(d: dict):
            tupla = []
            for k, _ in keys:
                v = _get_path(d, k)
                tupla.append((0, v) if v is not None else (1, None))
            return tupla

        self._docs.sort(key=_clave)
        if any(dir_ == -1 for _, dir_ in keys):
            self._docs.reverse()
        return self

    def limit(self, n: int) -> "FakeCursor":
        self._docs = self._docs[:n]
        return self

    def skip(self, n: int) -> "FakeCursor":
        self._docs = self._docs[n:]
        return self

    async def to_list(self, length: int | None = None) -> list[dict]:
        return list(self._docs[:length] if length is not None else self._docs)


class FakeCollection:
    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.docs: list[dict] = []

    def find(self, filtro: dict | None = None) -> FakeCursor:
        return FakeCursor([d for d in self.docs if _match(d, filtro or {})])

    async def find_one(self, filtro: dict | None = None) -> dict | None:
        for d in self.docs:
            if _match(d, filtro or {}):
                return dict(d)
        return None

    async def insert_one(self, doc: dict) -> SimpleNamespace:
        nuevo = dict(doc)
        nuevo.setdefault("_id", ObjectId())
        self.docs.append(nuevo)
        return SimpleNamespace(inserted_id=nuevo["_id"])

    async def update_one(self, filtro: dict, update: dict, upsert: bool = False) -> SimpleNamespace:
        for d in self.docs:
            if _match(d, filtro):
                for k, v in (update.get("$set") or {}).items():
                    _set_path(d, k, v)
                return SimpleNamespace(modified_count=1, upserted_id=None)
        if upsert:
            nuevo = {k: v for k, v in filtro.items() if not isinstance(v, dict)}
            nuevo.update(update.get("$set") or {})
            nuevo.setdefault("_id", ObjectId())
            self.docs.append(nuevo)
            return SimpleNamespace(modified_count=0, upserted_id=nuevo["_id"])
        return SimpleNamespace(modified_count=0, upserted_id=None)

    async def update_many(self, filtro: dict, update: dict) -> SimpleNamespace:
        n = 0
        for d in self.docs:
            if _match(d, filtro):
                for k, v in (update.get("$set") or {}).items():
                    _set_path(d, k, v)
                n += 1
        return SimpleNamespace(modified_count=n)

    async def delete_many(self, filtro: dict) -> SimpleNamespace:
        antes = len(self.docs)
        self.docs = [d for d in self.docs if not _match(d, filtro)]
        return SimpleNamespace(deleted_count=antes - len(self.docs))

    async def count_documents(self, filtro: dict | None = None) -> int:
        return sum(1 for d in self.docs if _match(d, filtro or {}))

    async def create_index(self, *args: Any, **kwargs: Any):
        return None


class FakeMongo:
    """Misma superficie que app.db.mongo.Mongo (colecciones fuente + propias del ML)."""

    def __init__(self) -> None:
        self.lecturas = FakeCollection("lecturas")
        self.eventos = FakeCollection("eventos")
        self.pacientes = FakeCollection("pacientes")
        self.medicamentos = FakeCollection("medicamentos")
        self.features = FakeCollection("features")
        self.modelos = FakeCollection("modelos")
        self.predicciones = FakeCollection("predicciones")
        self.eventos_confirmados = FakeCollection("eventos_confirmados")

    @property
    def ml(self) -> SimpleNamespace:
        mapeo = {
            "features_pacientes": self.features,
            "predicciones": self.predicciones,
            "eventos_confirmados": self.eventos_confirmados,
        }
        return SimpleNamespace(get_collection=lambda n: mapeo.get(n, FakeCollection(n)))

    @property
    def client(self):
        return None


def sembrar_paciente(
    fake: FakeMongo,
    paciente_id: str,
    n_lecturas: int = 60,
    fc: int = 72,
    temp: float = 36.6,
    gsr: float = 2.0,
    base: datetime | None = None,
    cadencia_min: int = 5,
) -> datetime:
    """Siembra un paciente con lecturas cada `cadencia_min` y devuelve su timestamp base."""
    base = base or (datetime.now(timezone.utc) - timedelta(days=30))
    fake.pacientes.docs.append(
        {
            "_id": paciente_id,
            "biometria": {"edad": 45, "peso_kg": 80, "estatura_cm": 175, "es_diabetico": True},
            "fecha_registro": base - timedelta(days=200),
        }
    )
    for i in range(n_lecturas):
        fake.lecturas.docs.append(
            {
                "meta": {"paciente_id": paciente_id},
                "timestamp": base + timedelta(minutes=cadencia_min * i),
                "pulso_bpm": fc,
                "temperatura_c": temp,
                "sudoracion_gsr": gsr,
                "hrv": 55.0,
                "spo2": 97,
            }
        )
    return base


@pytest.fixture
def fake_mongo() -> FakeMongo:
    return FakeMongo()


@pytest.fixture
def client(fake_mongo: FakeMongo):
    return TestClient(build_app(mongo=fake_mongo))


@pytest.fixture
def client_con_db(fake_mongo: FakeMongo):
    s = Settings(mongo_uri="mongodb://fake", worker_sync_enabled=False)
    return TestClient(build_app(mongo=fake_mongo, settings_override=s))
