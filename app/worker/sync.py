"""Worker de sincronización: consume Change Streams de lecturas_sensores y
mantiene features_pacientes (bioguard_ml) con TTL. Con fallback a polling.

Uso: python -m app.worker.sync
"""

from __future__ import annotations

import asyncio
import logging

from pymongo.errors import OperationFailure, PyMongoError

from app.core.config import get_settings
from app.db.mongo import Mongo
from app.models.features import normalizar_ts
from app.services.predictor import PredictorService

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEG = 30


async def _procesar_paciente(mongo: Mongo, predictor: PredictorService, paciente_id: str) -> None:
    from datetime import datetime, timedelta, timezone

    feats = await predictor.construir_features_db(paciente_id)
    if not feats:
        return
    settings = get_settings()
    ahora = datetime.now(timezone.utc)
    doc = {
        "paciente_id": paciente_id,
        "features": feats,
        "timestamp": ahora,
        "expireAt": ahora + timedelta(days=settings.features_ttl_days),
    }
    await mongo.features.update_one({"paciente_id": paciente_id}, {"$set": doc}, upsert=True)
    logger.info("Features actualizadas para %s", paciente_id)


async def _cambios(mongo: Mongo, predictor: PredictorService) -> None:
    async with mongo.lecturas.watch(full_document="updateLookup") as stream:
        async for change in stream:
            doc = change.get("fullDocument")
            if not doc:
                continue
            doc = normalizar_ts(doc)
            paciente_id = (doc.get("meta") or {}).get("paciente_id")
            if paciente_id:
                await _procesar_paciente(mongo, predictor, paciente_id)


async def _polling(mongo: Mongo, predictor: PredictorService) -> None:
    from datetime import datetime, timedelta, timezone

    while True:
        try:
            desde = datetime.now(timezone.utc) - timedelta(hours=1)
            docs = (
                await mongo.lecturas.find({"timestamp": {"$gte": desde}})
                .sort("timestamp", -1)
                .limit(200)
                .to_list(200)
            )
            pacientes = {d.get("meta", {}).get("paciente_id") for d in docs if d.get("meta")}
            for pid in pacientes:
                if pid:
                    await _procesar_paciente(mongo, predictor, pid)
        except Exception:
            logger.exception("Error en ciclo de polling")
        await asyncio.sleep(POLL_INTERVAL_SEG)


async def run_sync() -> None:
    settings = get_settings()
    mongo = Mongo(settings)
    await mongo.connect()
    predictor = PredictorService(mongo, settings)
    logger.info("Worker de sincronización iniciado (db_ml=%s)", settings.ml_db)
    try:
        await _cambios(mongo, predictor)
    except (OperationFailure, PyMongoError):
        logger.warning("Change Streams no disponible; usando modo polling")
        await _polling(mongo, predictor)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run_sync())


if __name__ == "__main__":
    main()
