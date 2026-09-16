from typing import Optional

from fastapi import APIRouter

from agents.tools import check_inventory

router = APIRouter()


@router.get("/api/inventory")
def inventory(blood_type: Optional[str] = None):
    return check_inventory(blood_type)
