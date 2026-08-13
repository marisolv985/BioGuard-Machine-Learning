"""Orquestador de predicción: une Mongo (fuente), features, baseline v1 y modelo v2."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId

from app.core.config import Settings
from app.db.mongo import Mongo
from app.models.features import construir_features, normalizar_ts
from app.models.inferencia import ModeloActivo, construir_resultado_v2
from app.models.entrenamiento import deserializar_artefacto
from app.models.pico_glucemico import PesosPico, evaluar
from app.schemas.prediccion import PrediccionRespuesta
from app.schemas.telemetria import TelemetriaEntrada
from app.services.baseline import predecir_baseline

logger = logging.getLogger(__name__)


def _filtro_id(paciente_id: str) -> dict[str, Any]:
    try:
        return {"_id": ObjectId(paciente_id)}
    except Exception:
        return {"_id": paciente_id}


class PredictorService:
    def __init__(self, mongo: Mongo, settings: Settings, modelo: ModeloActivo | None = None) -> None:
        self._mongo = mongo
        self._settings = settings
        self._modelo = modelo

    async def invalidar_modelo(self) -> None:
        self._modelo = None

    async def modelo_activo(self) -> ModeloActivo:
        if self._modelo is None:
            self._modelo = await self._cargar_modelo()
        return self._modelo

    async def _cargar_modelo(self) -> ModeloActivo:
        try:
            doc = await self._mongo.modelos.find_one({"activo": True})
        except Exception:
            logger.exception("Fallo al cargar modelo activo desde Mongo")
            doc = None
        if not doc:
            return ModeloActivo(None, self._settings.umbral_critico)
        artefacto = deserializar_artefacto(doc.get("artefacto") or b"")
        return ModeloActivo(artefacto, self._settings.umbral_critico)

    async def _contexto_paciente(self, paciente_id: str) -> tuple[dict | None, dict | None]:
        try:
            paciente = await self._mongo.pacientes.find_one(_filtro_id(paciente_id))
            medicamento = await self._mongo.medicamentos.find_one(
                {"paciente_id": paciente_id, "activo": True}
            )
            return paciente, medicamento
        except Exception:
            logger.exception("Fallo al cargar contexto del paciente %s", paciente_id)
            return None, None

    async def _lecturas_ventana(self, paciente_id: str) -> list[dict]:
        try:
            docs = (
                await self._mongo.lecturas.find({"meta.paciente_id": paciente_id})
                .sort("timestamp", -1)
                .limit(90)
                .to_list(90)
            )
        except Exception:
            logger.exception("Fallo al leer lecturas de %s", paciente_id)
            return []
        docs = [normalizar_ts(d) for d in docs]
        docs.sort(key=lambda d: d["timestamp"])
        return docs

    async def _eventos_30d(self, paciente_id: str) -> list[dict]:
        desde = datetime.now(timezone.utc) - timedelta(days=30)
        try:
            docs = (
                await self._mongo.eventos.find({"paciente_id": paciente_id, "fecha_evento": {"$gte": desde}})
                .sort("fecha_evento", -1)
                .to_list(200)
            )
        except Exception:
            logger.exception("Fallo al leer eventos de %s", paciente_id)
            return []
        for d in docs:
            if d.get("fecha_evento") is not None and not hasattr(d["fecha_evento"], "tzinfo"):
                d["fecha_evento"] = d["fecha_evento"].replace(tzinfo=timezone.utc)
        return docs

    async def construir_features_db(self, paciente_id: str) -> dict[str, float] | None:
        docs = await self._lecturas_ventana(paciente_id)
        if len(docs) < 3:
            return None
        paciente, medicamento = await self._contexto_paciente(paciente_id)
        eventos = await self._eventos_30d(paciente_id)
        ventana = docs[-self._settings.window_size :]
        try:
            return construir_features(
                ventana,
                paciente=paciente,
                medicamento=medicamento,
                eventos_30d=eventos,
                baseline_extra=docs,
            )
        except (ValueError, KeyError):
            logger.exception("No fue posible construir features para %s", paciente_id)
            return None

    async def predecir_v1(self, telemetria: TelemetriaEntrada) -> PrediccionRespuesta:
        return predecir_baseline(telemetria, self._settings)

    async def predecir_v2(self, telemetria: TelemetriaEntrada) -> dict[str, Any]:
        features = None
        if self._settings.mongo_uri:
            features = await self.construir_features_db(telemetria.paciente_id)
        modelo = await self.modelo_activo()
        v1 = await self.predecir_v1(telemetria)
        return construir_resultado_v2(
            paciente_id=telemetria.paciente_id,
            telemetria=telemetria.model_dump(),
            features=features,
            modelo=modelo,
            probabilidad_general=v1.probabilidad,
            es_critico_general=v1.es_critico,
            version=self._settings.app_version,
        )

    async def pesos_pico(self) -> PesosPico:
        """Pesos de la red logística (F2) con fuente de verdad en Mongo.

        Lee el documento `{clave: <pesos_mongo_clave>}` de la colección `modelos`;
        si no existe, siembra los valores de Settings (fallback) y los persiste.
        """
        if not self._settings.mongo_uri:
            return PesosPico.from_settings(self._settings)
        try:
            doc = await self._mongo.modelos.find_one({"clave": self._settings.pesos_mongo_clave})
            if doc and doc.get("pesos"):
                return PesosPico.from_dict(doc["pesos"])
        except Exception:
            logger.exception("Fallo al cargar pesos de pico desde Mongo")
        semilla = PesosPico.from_settings(self._settings)
        try:
            await self._mongo.modelos.update_one(
                {"clave": self._settings.pesos_mongo_clave},
                {"$set": {"pesos": semilla.to_dict()}},
                upsert=True,
            )
            clave = self._settings.pesos_mongo_clave
            logger.info("Pesos de pico glucémico sembrados en Mongo (clave=%s)", clave)
        except Exception:
            logger.exception("Fallo al sembrar pesos de pico en Mongo")
        return semilla

    async def predecir_pico(self, telemetria: TelemetriaEntrada) -> dict[str, Any]:
        """Motor de picos glucémicos (F1/F2/F3 + matriz de riesgo)."""
        pesos = await self.pesos_pico()

        peso = telemetria.peso
        estatura = telemetria.estatura
        if peso is None or estatura is None:
            paciente, _ = await self._contexto_paciente(telemetria.paciente_id)
            biometria = (paciente or {}).get("biometria") or {}
            if peso is None:
                peso = biometria.get("peso_kg")
            if estatura is None:
                estatura_cm = biometria.get("estatura_cm") or biometria.get("estatura_m") or 0.0
                estatura = float(estatura_cm) / 100.0

        if peso is None or estatura is None or float(estatura) <= 0:
            raise ValueError(
                "no hay peso/estatura (request o biometría del paciente) para calcular el IMC"
            )

        resultado = evaluar(
            peso_kg=float(peso),
            estatura_m=float(estatura),
            pulso_bpm=float(telemetria.frecuencia_cardiaca),
            sudor_us=telemetria.sudoracion_gsr_normalizado,
            temperatura_c=float(telemetria.temperatura),
            pesos=pesos,
        )
        resultado["paciente_id"] = telemetria.paciente_id
        resultado["timestamp"] = datetime.now(timezone.utc)
        resultado["version"] = self._settings.app_version
        return resultado
