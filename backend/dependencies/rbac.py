"""
Role-based access control.

Role permission matrix (from the project plan):

  Admin              -- full access
  Blood Bank Manager  -- inventory, forecast, donor alerts, prediction
  Hospital Staff       -- patient entry (all formats), prediction/explanation,
                           inventory read-only, chat
  Donor Coordinator     -- donor management + alerts only
  Auditor                -- read-only: audit log, predictions, forecasts

`get_current_user` decodes the bearer token and loads the User row.
`require_role(*roles)` is a dependency factory -- attach it to any
route to reject any caller whose role isn't in the given set. "Admin"
always passes, regardless of which roles are listed, since Admin means
full access everywhere.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from backend.services.auth import decode_access_token
from db.database import get_db
from db.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        email = payload.get("sub")
        if email is None:
            raise credentials_error
    except JWTError:
        raise credentials_error

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_error
    return user


def require_role(*roles: str):
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != "Admin" and current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not permitted to access this resource.",
            )
        return current_user
    return dependency
