import json

from fastapi import APIRouter, Depends

from agents.tools import get_blood_demand_forecast
from backend.dependencies.rbac import require_role
from backend.services.cache import cache_get, cache_set

router = APIRouter()

FORECAST_CACHE_TTL_SECONDS = 60 * 60  # 1 hour


@router.get("/api/forecast", dependencies=[Depends(require_role("Blood Bank Manager", "Auditor"))])
def forecast(days: int = 7):
    cache_key = f"forecast:{days}"

    cached = cache_get(cache_key)
    if cached:
        result = json.loads(cached)
        result["cached"] = True
        return result

    result = get_blood_demand_forecast(days)
    if "error" not in result:
        cache_set(cache_key, json.dumps(result), FORECAST_CACHE_TTL_SECONDS)
    result["cached"] = False
    return result
