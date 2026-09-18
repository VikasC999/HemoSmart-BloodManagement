from typing import Optional

from fastapi import APIRouter, Depends

from agents.tools import check_inventory
from backend.dependencies.rbac import require_role

router = APIRouter()


@router.get("/api/inventory", dependencies=[Depends(require_role("Hospital Staff", "Blood Bank Manager"))])
def inventory(blood_type: Optional[str] = None):
    return check_inventory(blood_type)
