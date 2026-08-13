import logging

from fastapi import APIRouter, Depends, Request, status

from app.api.deps import auth_servicio, get_dashboard, get_predictor
from app.core.limiter import limiter
from app.schemas.prediccion import PicoGlucemicoRespuesta, PrediccionRespuesta, PrediccionV2
from app.schemas.telemetria import TelemetriaEntrada
from app.services.dashboard import DashboardService
from app.services.predictor import PredictorService

logger = logging.getLogger(__name__)

v1_router = APIRouter(prefix="/api/v1", tags=["predicciones-v1"])
v2_router = APIRouter(prefix="/api/v2", tags=["predicciones-v2"])
v3_router = APIRouter(prefix="/api/v3", tags=["picos-glucemicos"])


@v1_router.post(
    "/predicciones",
    response_model=PrediccionRespuesta,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_servicio)],
    summary="Predicción v1 (contrato legado)",
)
async def prediccion_v1(
    telemetria: TelemetriaEntrada,
    predictor: PredictorService = Depends(get_predictor),
) -> PrediccionRespuesta:
    return await predictor.predecir_v1(telemetria)


@limiter.limit("100/minute")
@v2_router.post(
    "/predicciones",
    response_model=PrediccionV2,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_servicio)],
    summary="Predicción v2 con distinción hipo/hiperglucemia",
)
async def prediccion_v2(
    request: Request,
    telemetria: TelemetriaEntrada,
    predictor: PredictorService = Depends(get_predictor),
    dashboard: DashboardService = Depends(get_dashboard),
) -> PrediccionV2:
    resultado = await predictor.predecir_v2(telemetria)
    try:
        await dashboard.guardar_prediccion(telemetria.paciente_id, resultado)
    except Exception:
        logger.exception("No se pudo persistir la predicción para %s", telemetria.paciente_id)
    return PrediccionV2.model_validate(resultado)


@limiter.limit("100/minute")
@v3_router.post(
    "/predicciones",
    response_model=PicoGlucemicoRespuesta,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_servicio)],
    summary="Picos glucémicos (F1 IMC, F2 z, F3 P(Pico)) + matriz de riesgo",
)
async def pico_glucemico(
    request: Request,
    telemetria: TelemetriaEntrada,
    predictor: PredictorService = Depends(get_predictor),
) -> PicoGlucemicoRespuesta:
    return PicoGlucemicoRespuesta.model_validate(await predictor.predecir_pico(telemetria))
