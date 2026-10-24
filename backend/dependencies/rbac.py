"""
Role-based access control.

The role permission matrix lives in schema/permissions.py, shared with
the chat assistant so the two can't drift (D65).

`get_current_user` decodes the bearer token and loads the User row.
`require_permission(capability)` is the dependency routes attach: it
rejects any caller whose role doesn't hold that capability. Admin holds
every capability. `require_role(*roles)` is the older, role-list form,
kept for tests and ad-hoc guards.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from backend.services.auth import decode_access_token
from db.database import get_db
from db.models import User
from schema.permissions import denial, is_permitted

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


def require_permission(capability: str):
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if not is_permitted(current_user.role, capability):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=denial(current_user.role, capability),
            )
        return current_user
    dependency.capability = capability  # read by the route/matrix parity test
    return dependency
