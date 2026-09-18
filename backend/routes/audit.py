from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.dependencies.rbac import require_role
from db.database import get_db
from db.models import AuditLog

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

    class Config:
        from_attributes = True


@router.get(
    "/api/audit-log",
    response_model=List[AuditLogOut],
    dependencies=[Depends(require_role("Auditor"))],
)
def list_audit_log(limit: int = 100, db: Session = Depends(get_db)):
    """Most recent entries first. Auditor/Admin only, per the RBAC matrix."""
    rows = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return [
        AuditLogOut(
            id=r.id,
            user_id=r.user_id,
            action=r.action,
            resource_type=r.resource_type,
            resource_id=r.resource_id,
            ip_address=r.ip_address,
            timestamp=r.timestamp.isoformat(),
            details=r.details,
        )
        for r in rows
    ]
