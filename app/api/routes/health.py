from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.db.mongo import Mongo
from app.services.predictor import PredictorService

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
async def health(request: Request) -> dict:
    predictor: PredictorService = request.app.state.predictor
    modelo_id = "baseline-v0"
    if settings.mongo_uri:
        try:
            modelo = await predictor.modelo_activo()
            modelo_id = modelo.version or "baseline-v0"
        except Exception:
            modelo_id = "no_disponible"
    return {
        "estado": "ok",
        "servicio": "BioGuard ML Service",
        "version": settings.app_version,
        "modeloActivo": modelo_id,
        "umbralCritico": settings.umbral_critico,
        "enviroment": settings.app_env,
    }


@router.get("/ready")
async def ready(request: Request) -> dict:
    mongo: Mongo = request.app.state.mongo
    mongo_ok = False
    if settings.mongo_uri:
        try:
            await mongo.client.admin.command("ping")
            mongo_ok = True
        except Exception:
            mongo_ok = False
    return {"estado": "listo" if mongo_ok or not settings.mongo_uri else "no_listo", "mongo": mongo_ok}
