"""Agregaciones orientadas a dashboard: riesgo actual, tendencias, historial, resumen."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId

from app.db.mongo import Mongo
from app.schemas.dashboard import (
    HistorialItem,
    HistorialPage,
    PuntoTendencia,
    RangoTendencia,
    ResumenDashboard,
    EventoConfirmacionRequest,
)

_RANGO_DIAS = {
    RangoTendencia.DIA: 1,
    RangoTendencia.SEMANA: 7,
    RangoTendencia.MES: 30,
}


class DashboardService:
    def __init__(self, mongo: Mongo) -> None:
        self._mongo = mongo

    async def guardar_prediccion(self, paciente_id: str, resultado: dict[str, Any]) -> str | None:
        doc = {
            "paciente_id": paciente_id,
            "riesgo": resultado["riesgo"],
            "explicacion": resultado.get("explicacion", []),
            "recomendacion": resultado.get("recomendacion", ""),
            "modelo": resultado.get("modelo", {}),
            "timestamp": resultado.get("timestamp", datetime.now(timezone.utc)),
        }
        res = await self._mongo.predicciones.insert_one(doc)
        return str(res.inserted_id)

    async def riesgo_actual(self, paciente_id: str) -> dict[str, Any] | None:
        doc = await self._ultima(paciente_id)
        if not doc:
            return None
        return doc.get("riesgo")

    async def _ultima(self, paciente_id: str) -> dict[str, Any] | None:
        docs = (
            await self._mongo.predicciones.find({"paciente_id": paciente_id})
            .sort("timestamp", -1)
            .limit(1)
            .to_list(1)
        )
        return docs[0] if docs else None

    async def tendencias(self, paciente_id: str, rango: RangoTendencia) -> list[PuntoTendencia]:
        dias = _RANGO_DIAS[rango]
        desde = datetime.now(timezone.utc) - timedelta(days=dias)
        docs = (
            await self._mongo.predicciones.find({"paciente_id": paciente_id, "timestamp": {"$gte": desde}})
            .sort("timestamp", 1)
            .to_list(2000)
        )
        bucket_tam: timedelta = timedelta(hours=1) if rango == RangoTendencia.DIA else timedelta(days=1)

        agrupado: dict[datetime, list[float]] = {}
        picos: dict[datetime, float] = {}
        for d in docs:
            ts = d.get("timestamp")
            if ts is None:
                continue
            if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if rango == RangoTendencia.DIA:
                bucket = ts.replace(minute=0, second=0, microsecond=0)
            else:
                bucket = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            proba = (d.get("riesgo") or {}).get("probabilidad", 0.0)
            agrupado.setdefault(bucket, []).append(proba)
            picos[bucket] = max(picos.get(bucket, 0.0), proba)

        puntos: list[PuntoTendencia] = []
        cursor = desde
        while cursor <= datetime.now(timezone.utc):
            if rango == RangoTendencia.DIA:
                bucket = cursor.replace(minute=0, second=0, microsecond=0)
            else:
                bucket = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
            valores = agrupado.get(bucket, [])
            if valores:
                puntos.append(
                    PuntoTendencia(
                        fecha=bucket,
                        riesgo_promedio=round(sum(valores) / len(valores), 4),
                        probabilidad_pico=round(picos.get(bucket, 0.0), 4),
                        n_predicciones=len(valores),
                    )
                )
            cursor += bucket_tam
        return puntos

    async def historial(self, paciente_id: str, pagina: int, pagina_tamano: int) -> HistorialPage:
        filtro = {"paciente_id": paciente_id}
        total = await self._mongo.predicciones.count_documents(filtro)
        skip = (pagina - 1) * pagina_tamano
        docs = (
            await self._mongo.predicciones.find(filtro)
            .sort("timestamp", -1)
            .skip(skip)
            .limit(pagina_tamano)
            .to_list(pagina_tamano)
        )
        items = [
            HistorialItem(
                prediccion_id=str(d.get("_id")),
                probabilidad=((d.get("riesgo") or {}).get("probabilidad") or 0.0),
                tipo=(d.get("riesgo") or {}).get("tipo", "normal"),
                nivel_riesgo=(d.get("riesgo") or {}).get("tipo", "normal"),
                timestamp=d.get("timestamp", datetime.now(timezone.utc)),
                es_critico=bool((d.get("riesgo") or {}).get("es_critico", False)),
            )
            for d in docs
        ]
        return HistorialPage(items=items, total=total, pagina=pagina, pagina_tamano=pagina_tamano)

    async def resumen(self, paciente_id: str, rango: RangoTendencia) -> ResumenDashboard:
        ultima = await self._ultima(paciente_id)
        tendencia = await self.tendencias(paciente_id, rango)
        riesgo = (ultima or {}).get("riesgo")
        return ResumenDashboard(
            paciente_id=paciente_id,
            riesgo_actual=riesgo,
            explicacion=(ultima or {}).get("explicacion", []),
            recomendacion=(ultima or {}).get("recomendacion", ""),
            tendencia=tendencia,
            modelo=(ultima or {}).get("modelo"),
            ultima_actualizacion=(ultima or {}).get("timestamp", datetime.now(timezone.utc)),
        )

    async def explicabilidad(self, prediccion_id: str) -> list[dict[str, Any]]:
        try:
            _id = ObjectId(prediccion_id)
        except Exception:
            _id = prediccion_id
        doc = await self._mongo.predicciones.find_one({"_id": _id})
        if not doc:
            return []
        return doc.get("explicacion", [])

    async def confirmar_evento(self, req: EventoConfirmacionRequest) -> None:
        await self._mongo.eventos_confirmados.insert_one(
            {
                "paciente_id": req.paciente_id,
                "evento_id": req.evento_id,
                "confirmado": req.confirmado,
                "nota": req.nota,
                "fecha_confirmacion": datetime.now(timezone.utc),
                "origen": "servicio",
            }
        )
