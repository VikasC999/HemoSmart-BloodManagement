from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from agents.tools import DEFAULT_INVENTORY, _load_inventory, _save_inventory, check_inventory
from backend.dependencies.rbac import require_role

router = APIRouter()


class InventoryUpdateRequest(BaseModel):
    blood_type: str
    units: int = Field(..., ge=0)


@router.get("/api/inventory", dependencies=[Depends(require_role("Hospital Staff", "Blood Bank Manager"))])
def inventory(blood_type: Optional[str] = None):
    return check_inventory(blood_type)


@router.patch("/api/inventory", dependencies=[Depends(require_role("Blood Bank Manager"))])
def update_inventory(payload: InventoryUpdateRequest, request: Request):
    """Sets a blood type's stock level directly -- e.g. after a physical
    stock count, or recording units used/received. Blood Bank Manager
    only; Hospital Staff's inventory access stays read-only per the
    RBAC matrix.
    """
    if payload.blood_type not in DEFAULT_INVENTORY:
        raise HTTPException(status_code=422, detail=f"Unknown blood type: {payload.blood_type}")

    current = _load_inventory()
    previous_units = current.get(payload.blood_type)
    current[payload.blood_type] = payload.units
    _save_inventory(current)

    request.state.audit_action = "inventory_changed"
    request.state.audit_resource_type = "inventory"
    request.state.audit_resource_id = payload.blood_type
    request.state.audit_details = {
        "blood_type": payload.blood_type,
        "previous_units": previous_units,
        "new_units": payload.units,
    }

    return {"blood_type": payload.blood_type, "units": payload.units}
