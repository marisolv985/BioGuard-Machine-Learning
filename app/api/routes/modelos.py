import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request

from app.api.deps import auth_servicio, get_mongo, get_predictor, get_retrain
from app.core.limiter import limiter
from app.models.drift import calcular_drift
from app.models.features import FEATURES
from app.schemas.admin import EntrenarRequest, ModeloActivoResponse, ModeloRegistro
from app.services.predictor import PredictorService
from app.services.retrain import RetrainService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2",
    tags=["modelos"],
    dependencies=[Depends(auth_servicio)],
)


async def _doc_activo(mongo) -> Optional[dict]:
    return await mongo.modelos.find_one({"activo": True})


@router.get("/modelos/activo", response_model=ModeloActivoResponse)
async def modelo_activo(
    predictor: PredictorService = Depends(get_predictor),
    mongo=Depends(get_mongo),
) -> ModeloActivoResponse:
    doc = await _doc_activo(mongo)
    modelo = await predictor.modelo_activo()

    drift = None
    if doc:
        try:
            recientes = (
                await mongo.features.find({}).sort("timestamp", -1).limit(100).to_list(100)
            )
            if recientes:
                features_list = [
                    {f: (r.get("features") or {}).get(f) for f in FEATURES} for r in recientes
                ]
                artefacto = modelo.artefacto
                if artefacto:
                    drift = calcular_drift(features_list, artefacto.get("referencia", {}))
        except Exception:
            logger.warning("No fue posible calcular el drift", exc_info=True)

    if not doc:
        return ModeloActivoResponse(
            version=modelo.version or "sin-modelo",
            tipo="fallback-heuristico",
            activo=False,
            total_muestras=0,
            fecha_entrenamiento=None,
            features=FEATURES,
            drift=drift,
        )

    return ModeloActivoResponse(
        version=doc.get("version", ""),
        tipo=doc.get("tipo", ""),
        activo=bool(doc.get("activo")),
        metricas=doc.get("metricas"),
        total_muestras=doc.get("total_muestras", 0),
        fecha_entrenamiento=doc.get("fecha_entrenamiento"),
        features=doc.get("features") or FEATURES,
        drift=drift,
    )


@limiter.limit("5/minute")
@router.post("/modelos/entrenar", response_model=Optional[ModeloRegistro])
async def entrenar(
    request: Request,
    payload: EntrenarRequest,
    retrain: RetrainService = Depends(get_retrain),
    predictor: PredictorService = Depends(get_predictor),
) -> Optional[ModeloRegistro]:
    doc = await retrain.entrenar(descripcion=payload.descripcion, retener_activo=payload.retener_activo)
    if doc is None:
        return None
    await predictor.invalidar_modelo()
    return ModeloRegistro(
        id=doc["id"],
        version=doc["version"],
        tipo=doc["tipo"],
        activo=bool(doc["activo"]),
        total_muestras=doc["total_muestras"],
        fecha_entrenamiento=doc["fecha_entrenamiento"],
        descripcion=doc.get("descripcion"),
        metricas=doc.get("metricas"),
        conteos=doc.get("conteos"),
    )


@router.get("/modelos", response_model=list[ModeloRegistro])
async def listar_modelos(mongo=Depends(get_mongo)) -> list[ModeloRegistro]:
    docs = await mongo.modelos.find({}).sort("fecha_entrenamiento", -1).to_list(50)
    return [
        ModeloRegistro(
            id=str(d.get("_id")),
            version=d.get("version", ""),
            tipo=d.get("tipo", ""),
            activo=bool(d.get("activo")),
            total_muestras=d.get("total_muestras", 0),
            fecha_entrenamiento=d.get("fecha_entrenamiento"),
            descripcion=d.get("descripcion"),
            metricas=d.get("metricas"),
            conteos=d.get("conteos"),
        )
        for d in docs
    ]
