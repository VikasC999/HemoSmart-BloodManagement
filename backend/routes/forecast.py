from fastapi import APIRouter

from agents.tools import get_blood_demand_forecast

router = APIRouter()


@router.get("/api/forecast")
def forecast(days: int = 7):
    return get_blood_demand_forecast(days)
