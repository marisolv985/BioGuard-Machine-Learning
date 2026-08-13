"""Job de reentrenamiento (cron): colecta datos, entrena y registra el modelo.

Uso: python -m app.worker.retrain [--descripcion "..." ] [--promover]
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.core.config import get_settings
from app.db.mongo import Mongo
from app.services.retrain import RetrainService

logger = logging.getLogger(__name__)


async def run_retrain(descripcion: str | None, retener_activo: bool) -> None:
    settings = get_settings()
    mongo = Mongo(settings)
    await mongo.connect()
    servicio = RetrainService(mongo, settings)
    resultado = await servicio.entrenar(descripcion=descripcion, retener_activo=retener_activo)
    if resultado:
        logger.info(
            "Entrenamiento completado: version=%s muestras=%s metricas=%s",
            resultado["version"],
            resultado["total_muestras"],
            resultado.get("metricas"),
        )
    else:
        logger.warning("Entrenamiento sin resultado (datos insuficientes)")
    await mongo.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Reentrena el modelo hipo/hiper")
    parser.add_argument("--descripcion", default=None)
    parser.add_argument("--promover", action="store_true", help="Activar el modelo recién entrenado")
    args = parser.parse_args()
    asyncio.run(run_retrain(args.descripcion, retener_activo=not args.promover))


if __name__ == "__main__":
    main()
