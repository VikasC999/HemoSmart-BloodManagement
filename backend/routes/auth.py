from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, EmailStr
from sqlalchemy.orm import Session

from backend.dependencies.rbac import get_current_user, require_role
from backend.services.auth import create_access_token, hash_password, verify_password
from db.database import get_db
from db.models import User

router = APIRouter()

VALID_ROLES = {"Admin", "Blood Bank Manager", "Hospital Staff", "Donor Coordinator", "Auditor"}


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    role: str = "Hospital Staff"


class UserOut(BaseModel):
    id: int
    email: str
    role: str

    model_config = ConfigDict(from_attributes=True)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/api/auth/register", response_model=UserOut, dependencies=[Depends(require_role("Admin"))])
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    # Admin-only: this endpoint assigns roles, including Admin itself,
    # so open self-registration would let anyone grant themselves full
    # access. Staff accounts are provisioned by an Admin.
    if payload.role not in VALID_ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {sorted(VALID_ROLES)}")

    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    request.state.audit_action = "user_created"
    request.state.audit_resource_type = "user"
    request.state.audit_resource_id = str(user.id)
    request.state.audit_details = {"email": user.email, "role": user.role}
    return user


@router.post("/api/auth/login", response_model=TokenOut)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm's "username" field carries the email --
    # this shape is what lets FastAPI's /docs "Authorize" button work
    # out of the box, without a custom login form there.
    user = db.query(User).filter(User.email == form_data.username).first()
    if user is None or user.hashed_password is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = create_access_token(subject=user.email, role=user.role)
    return TokenOut(access_token=token)


@router.get("/api/auth/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user
