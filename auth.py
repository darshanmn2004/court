# auth.py
import os
import json
import uuid
import time
from typing import Optional, Dict, Any
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import jwt

# ============================================================
# CONFIG
# ============================================================
USERS_FILE = Path("users.json")
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

JWT_SECRET = os.getenv("JWT_SECRET", "please_change_this_secret")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 24   # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

router = APIRouter(tags=["auth"])

_users_lock = Lock()


# ============================================================
# MODELS
# ============================================================
class SignupRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserPublic(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    role: Optional[str] = "user"
    created_at: Optional[int]


# ============================================================
# FILE HELPERS
# ============================================================
def _read_users() -> Dict[str, Dict[str, Any]]:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_users(data: Dict[str, Dict[str, Any]]):
    with _users_lock:
        USERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    users = _read_users()
    for uid, u in users.items():
        if u.get("email", "").lower() == email.lower():
            return {**u, "id": uid}
    return None


# ============================================================
# PASSWORD HELPERS (FIXED — bcrypt max length = 72 bytes)
# ============================================================
def _verify_password(plain: str, hashed: str) -> bool:
    plain = plain[:72]   # prevent bcrypt overflow
    return pwd_context.verify(plain, hashed)


def _hash_password(password: str) -> str:
    password = password[:72]   # prevent "password too long" crash
    return pwd_context.hash(password)


# ============================================================
# TOKEN HELPERS
# ============================================================
def _create_access_token(data: dict, expires_in=ACCESS_TOKEN_EXPIRE_SECONDS) -> str:
    payload = data.copy()
    payload.update({"exp": int(time.time()) + expires_in})

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token if isinstance(token, str) else token.decode("utf-8")


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# ============================================================
# CURRENT USER DEPENDENCY
# ============================================================
def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPublic:
    data = _decode_token(token)
    uid = data.get("sub")

    if uid is None:
        raise HTTPException(401, "Invalid token payload")

    users = _read_users()
    u = users.get(uid)

    if not u:
        raise HTTPException(401, "User not found")

    return UserPublic(
        id=uid,
        full_name=u.get("full_name"),
        email=u.get("email"),
        role=u.get("role", "user"),
        created_at=u.get("created_at")
    )


# ============================================================
# ROUTES
# ============================================================

@router.post("/signup", response_model=UserPublic, status_code=201)
def signup(req: SignupRequest):
    if _get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    uid = str(uuid.uuid4())
    user_obj = {
        "full_name": req.full_name,
        "email": req.email,
        "password_hash": _hash_password(req.password),
        "role": "user",
        "created_at": int(time.time())
    }

    users = _read_users()
    users[uid] = user_obj
    _write_users(users)

    return UserPublic(
        id=uid,
        full_name=req.full_name,
        email=req.email,
        role="user",
        created_at=user_obj["created_at"]
    )


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = _get_user_by_email(form_data.username)

    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    if not _verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    # Build JWT token
    payload = {
        "sub": user["id"],
        "email": user["email"],
        "role": user.get("role", "user")
    }

    token = _create_access_token(payload)

    return TokenResponse(access_token=token, expires_in=ACCESS_TOKEN_EXPIRE_SECONDS)


@router.get("/me", response_model=UserPublic)
def me(current_user: UserPublic = Depends(get_current_user)):
    return current_user
