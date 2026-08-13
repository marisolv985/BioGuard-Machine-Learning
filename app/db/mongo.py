import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Mongo:
    """Cliente Mongo compartido.

    - Datastore fuente (backend): acceso de solo lectura.
    - Base aislada del ML (bioguard_ml): acceso read/write.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncIOMotorClient | None = None
        self._source: AsyncIOMotorDatabase | None = None
        self._ml: AsyncIOMotorDatabase | None = None

    @property
    def client(self) -> AsyncIOMotorClient:
        if self._client is None:
            raise RuntimeError("Mongo no inicializado: llama a connect()")
        return self._client

    @property
    def source(self) -> AsyncIOMotorDatabase:
        if self._source is None:
            raise RuntimeError("Mongo no inicializado")
        return self._source

    @property
    def ml(self) -> AsyncIOMotorDatabase:
        if self._ml is None:
            raise RuntimeError("Mongo no inicializado")
        return self._ml

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = AsyncIOMotorClient(self._settings.mongo_uri, serverSelectionTimeoutMS=5000)
        self._source = self._client[self._settings.source_db]
        self._ml = self._client[self._settings.ml_db]
        logger.info("Mongo conectado: source=%s ml=%s", self._settings.source_db, self._settings.ml_db)

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._source = None
            self._ml = None

    # ---- Fuente (solo lectura) ----
    @property
    def lecturas(self) -> AsyncIOMotorCollection:
        return self.source["lecturas_sensores"]

    @property
    def eventos(self) -> AsyncIOMotorCollection:
        return self.source["eventos_metabolicos"]

    @property
    def pacientes(self) -> AsyncIOMotorCollection:
        return self.source["pacientes"]

    @property
    def medicamentos(self) -> AsyncIOMotorCollection:
        return self.source["medicamentos"]

    # ---- Propia del ML (read/write) ----
    @property
    def features(self) -> AsyncIOMotorCollection:
        return self.ml["features_pacientes"]

    @property
    def modelos(self) -> AsyncIOMotorCollection:
        return self.ml["modelos"]

    @property
    def predicciones(self) -> AsyncIOMotorCollection:
        return self.ml["predicciones"]

    @property
    def eventos_confirmados(self) -> AsyncIOMotorCollection:
        return self.ml["eventos_confirmados"]
