from fastapi import APIRouter, Depends, Request, status

from app.api.deps import auth_servicio, get_dashboard, get_mongo
from app.core.limiter import limiter
from app.schemas.dashboard import EventoConfirmacionRequest
from app.services.dashboard import DashboardService

router = APIRouter(
    prefix="/api/internal",
    tags=["internal"],
    dependencies=[Depends(auth_servicio)],
)


@limiter.limit("30/minute")
@router.post("/eventos/confirmar", status_code=status.HTTP_204_NO_CONTENT)
async def confirmar_evento(
    request: Request,
    payload: EventoConfirmacionRequest,
    dashboard: DashboardService = Depends(get_dashboard),
) -> None:
    await dashboard.confirmar_evento(payload)


@router.delete("/pacientes/{paciente_id}", status_code=status.HTTP_204_NO_CONTENT)
async def purgar_paciente(paciente_id: str, mongo=Depends(get_mongo)) -> None:
    """Derecho al olvido: purga todo dato derivado del paciente en bioguard_ml."""
    for nombre in ("features_pacientes", "predicciones", "eventos_confirmados"):
        await mongo.ml.get_collection(nombre).delete_many({"paciente_id": paciente_id})
