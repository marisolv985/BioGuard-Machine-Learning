"""Job de reentrenamiento: colecta datos del datastore fuente, construye el
dataset clínico y registra el nuevo modelo (con gate de calidad)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import Binary

from app.core.config import Settings
from app.db.mongo import Mongo
from app.models.entrenamiento import entrenar_modelos, serializar_artefacto
from app.models.features import FEATURES, normalizar_ts
from app.models.labels import preparar_dataset

logger = logging.getLogger(__name__)

LECTURAS_DIAS = 60
EVENTOS_DIAS = 90
MAX_PACIENTES = 500


def _aware(ts: Any) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


class RetrainService:
    def __init__(self, mongo: Mongo, settings: Settings) -> None:
        self._mongo = mongo
        self._settings = settings

    async def _lecturas(self, paciente_id: str) -> list[dict]:
        desde = datetime.now(timezone.utc) - timedelta(days=LECTURAS_DIAS)
        docs = (
            await self._mongo.lecturas.find(
                {"meta.paciente_id": paciente_id, "timestamp": {"$gte": desde}}
            )
            .sort("timestamp", 1)
            .to_list(2000)
        )
        return [normalizar_ts(d) for d in docs]

    async def _eventos(self, paciente_id: str) -> list[dict]:
        desde = datetime.now(timezone.utc) - timedelta(days=EVENTOS_DIAS)
        docs = (
            await self._mongo.eventos.find({"paciente_id": paciente_id, "fecha_evento": {"$gte": desde}})
            .sort("fecha_evento", 1)
            .to_list(500)
        )
        for d in docs:
            d["fecha_evento"] = _aware(d.get("fecha_evento"))
        return docs

    async def _colectar(self) -> dict[str, dict]:
        datos: dict[str, dict] = {}
        pacientes = await self._mongo.pacientes.find({}).limit(MAX_PACIENTES).to_list(MAX_PACIENTES)
        for pac in pacientes:
            pid = str(pac.get("_id"))
            lecturas = await self._lecturas(pid)
            if len(lecturas) < 5:
                continue
            if pac.get("fecha_registro") is not None:
                pac["fecha_registro"] = _aware(pac["fecha_registro"])
            eventos = await self._eventos(pid)
            medicamento = await self._mongo.medicamentos.find_one({"paciente_id": pid, "activo": True})
            datos[pid] = {
                "paciente": pac,
                "lecturas": lecturas,
                "eventos": eventos,
                "medicamento": medicamento,
            }
        return datos

    async def entrenar(
        self,
        descripcion: str | None = None,
        retener_activo: bool = True,
    ) -> dict[str, Any] | None:
        datos = await self._colectar()
        if not datos:
            logger.warning("No hay datos fuente para entrenar")
            return None
        ds = preparar_dataset(datos)
        if not ds["y"]:
            logger.warning("El dataset resultante está vacío (conteos=%s)", ds["conteos"])
            return None
        logger.info("Dataset: %s", ds["conteos"])
        if ds["conteos"]["normal"] < self._settings.modelo_min_muestras:
            logger.warning(
                "Muestras normales insuficientes (%s < %s); no se entrena",
                ds["conteos"]["normal"],
                self._settings.modelo_min_muestras,
            )
            return None

        artefacto = entrenar_modelos(ds["X"], ds["y"])
        if artefacto is None:
            logger.warning("Entrenamiento no produjo artefacto")
            return None

        doc: dict[str, Any] = {
            "version": artefacto["version"],
            "tipo": artefacto["tipo"],
            "fecha_entrenamiento": artefacto["fecha_entrenamiento"],
            "total_muestras": artefacto["total_muestras"],
            "conteos": artefacto["conteos"],
            "metricas": artefacto["metricas"],
            "features": FEATURES,
            "descripcion": descripcion,
            "activo": not retener_activo,
            "artefacto": Binary(serializar_artefacto(artefacto)),
        }
        if doc["activo"]:
            await self._mongo.modelos.update_many({"activo": True}, {"$set": {"activo": False}})
        res = await self._mongo.modelos.insert_one(doc)
        logger.info("Modelo %s registrado (id=%s)", artefacto["version"], res.inserted_id)
        return {**doc, "id": str(res.inserted_id)}
