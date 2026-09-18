from fastapi import APIRouter, Depends

from agents.tools import get_blood_demand_forecast
from backend.dependencies.rbac import require_role

router = APIRouter()


@router.get("/api/forecast", dependencies=[Depends(require_role("Blood Bank Manager", "Auditor"))])
def forecast(days: int = 7):
    return get_blood_demand_forecast(days)
