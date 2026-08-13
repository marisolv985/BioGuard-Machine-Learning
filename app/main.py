import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import dashboard, health, internal, modelos, predict
from app.core.config import Settings, get_settings
from app.core.limiter import limiter
from app.db.mongo import Mongo
from app.services.dashboard import DashboardService
from app.services.predictor import PredictorService
from app.services.retrain import RetrainService

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


async def _crear_indices(mongo: Mongo) -> None:
    await mongo.features.create_index("expireAt", expireAfterSeconds=0)
    await mongo.predicciones.create_index([("paciente_id", 1), ("timestamp", -1)])
    await mongo.modelos.create_index("activo")
    await mongo.eventos_confirmados.create_index([("evento_id", 1), ("paciente_id", 1)], unique=True)


def _montar_estado(app: FastAPI, mongo: Mongo, s: Settings) -> None:
    app.state.settings = s
    app.state.mongo = mongo
    app.state.predictor = PredictorService(mongo, s)
    app.state.dashboard = DashboardService(mongo)
    app.state.retrain = RetrainService(mongo, s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo = Mongo(settings)
    if settings.mongo_uri:
        await mongo.connect()
        await _crear_indices(mongo)
    _montar_estado(app, mongo, settings)
    try:
        yield
    finally:
        await mongo.close()


def build_app(mongo: Mongo | None = None, settings_override: Settings | None = None) -> FastAPI:
    """Fábrica de la aplicación. Con `mongo` inyectado (tests) se omite el lifespan."""
    s = settings_override or settings
    m = mongo if mongo is not None else Mongo(s)
    app = FastAPI(
        title="BioGuard ML Service",
        description="Microservicio de ML/IA para predicción de riesgo metabólico "
        "(distinción hipo/hiperglucemia) sobre proxies fisiológicos.",
        version=s.app_version,
        lifespan=None if mongo is not None else lifespan,
    )
    _montar_estado(app, m, s)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    from app.core.middleware import SecurityHeadersMiddleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predict.v1_router)
    app.include_router(predict.v2_router)
    app.include_router(predict.v3_router)
    app.include_router(dashboard.router)
    app.include_router(modelos.router)
    app.include_router(internal.router)
    return app


app = build_app()
