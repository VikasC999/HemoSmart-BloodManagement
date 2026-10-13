import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from bloodbank.forecast import get_forecaster
from bloodbank.outlook import compute_outlook
from bloodbank.stock import utcnow
from backend.dependencies.rbac import require_role
from backend.services.cache import cache_get, cache_set
from db.database import get_db

router = APIRouter()

FORECAST_CACHE_TTL_SECONDS = 6 * 60 * 60


@router.get("/api/forecast", dependencies=[Depends(require_role("Blood Bank Manager", "Auditor"))])
def forecast(days: int = Query(7, ge=1, le=30), db: Session = Depends(get_db)):
    """Demand forecast per component plus the demand-supply gap per blood
    group x component: projected shortages and units projected to expire
    unused (see bloodbank/outlook.py).

    Only the forecast is cached (it depends on nothing but the date);
    the gap is recomputed against live stock on every call, so it's
    never stale after a transfusion or an autonomy cycle."""
    now = utcnow()
    cache_key = f"forecast:{now.date()}:{days}"

    cached = cache_get(cache_key)
    try:
        totals = json.loads(cached) if cached else get_forecaster().component_forecast(now.date(), days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Forecast model not trained: {exc}")
    if not cached:
        cache_set(cache_key, json.dumps(totals), FORECAST_CACHE_TTL_SECONDS)

    result = compute_outlook(db, now, days, totals=totals)
    result["cached_forecast"] = bool(cached)
    return result
