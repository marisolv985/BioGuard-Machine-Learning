from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import auth_servicio, get_dashboard
from app.schemas.dashboard import (
    HistorialPage,
    PuntoTendencia,
    RangoTendencia,
    ResumenDashboard,
)
from app.schemas.prediccion import AporteExplicativo, RiesgoActual
from app.services.dashboard import DashboardService

router = APIRouter(
    prefix="/api/v2",
    tags=["dashboard"],
    dependencies=[Depends(auth_servicio)],
)


@router.get("/pacientes/{paciente_id}/riesgo-actual", response_model=Optional[RiesgoActual])
async def riesgo_actual(
    paciente_id: str,
    dashboard: DashboardService = Depends(get_dashboard),
) -> Optional[RiesgoActual]:
    return await dashboard.riesgo_actual(paciente_id)


@router.get("/pacientes/{paciente_id}/tendencias", response_model=list[PuntoTendencia])
async def tendencias(
    paciente_id: str,
    rango: RangoTendencia = Query(default=RangoTendencia.SEMANA),
    dashboard: DashboardService = Depends(get_dashboard),
) -> list[PuntoTendencia]:
    return await dashboard.tendencias(paciente_id, rango)


@router.get("/pacientes/{paciente_id}/historial", response_model=HistorialPage)
async def historial(
    paciente_id: str,
    pagina: int = Query(default=1, ge=1),
    pagina_tamano: int = Query(default=20, ge=1, le=100, alias="paginaTamano"),
    dashboard: DashboardService = Depends(get_dashboard),
) -> HistorialPage:
    return await dashboard.historial(paciente_id, pagina, pagina_tamano)


@router.get("/pacientes/{paciente_id}/explicabilidad/{prediccion_id}", response_model=list[AporteExplicativo])
async def explicabilidad(
    paciente_id: str,
    prediccion_id: str,
    dashboard: DashboardService = Depends(get_dashboard),
) -> list[AporteExplicativo]:
    return await dashboard.explicabilidad(prediccion_id)


@router.get("/pacientes/{paciente_id}/resumen", response_model=ResumenDashboard)
async def resumen(
    paciente_id: str,
    rango: RangoTendencia = Query(default=RangoTendencia.SEMANA),
    dashboard: DashboardService = Depends(get_dashboard),
) -> ResumenDashboard:
    return await dashboard.resumen(paciente_id, rango)
