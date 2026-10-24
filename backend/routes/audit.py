from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from agents.tools import recent_audit_entries
from backend.dependencies.rbac import require_permission
from db.database import get_db

router = APIRouter()


class AuditLogOut(BaseModel):
    id: int
    user_id: Optional[int]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    ip_address: Optional[str]
    timestamp: str
    details: Optional[dict]

    model_config = ConfigDict(from_attributes=True)


@router.get(
    "/api/audit-log",
    response_model=List[AuditLogOut],
    dependencies=[Depends(require_permission("audit:read"))],
)
def list_audit_log(limit: int = 100, db: Session = Depends(get_db)):
    """Most recent entries first. Auditor/Admin only, per the RBAC matrix."""
    return recent_audit_entries(limit, db=db)
