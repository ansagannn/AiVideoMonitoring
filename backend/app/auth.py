from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel

Role = Literal["operator", "manager", "admin"]

TOKEN_TTL_SECONDS = int(os.getenv("AI_MONITOR_TOKEN_TTL", "14400"))

@dataclass
class StoredUser:
    username: str
    password: str
    role: Role

class User(BaseModel):
    username: str
    role: Role

class AuthLoginRequest(BaseModel):
    username: str
    password: str

class AuthLoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: str
    user: User

_USERS: dict[str, StoredUser] = {
    "operator": StoredUser(
        username="operator",
        password=os.getenv("AI_MONITOR_PASSWORD_OPERATOR", "operator123"),
        role="operator",
    ),
    "manager": StoredUser(
        username="manager",
        password=os.getenv("AI_MONITOR_PASSWORD_MANAGER", "manager123"),
        role="manager",
    ),
    "admin": StoredUser(
        username="admin",
        password=os.getenv("AI_MONITOR_PASSWORD_ADMIN", "admin123"),
        role="admin",
    ),
}

@dataclass
class TokenRecord:
    username: str
    role: Role
    expires_at: float

_SESSIONS: dict[str, TokenRecord] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_password(password: str, expected: str) -> bool:
    return secrets.compare_digest(password, expected)


def _get_user(username: str) -> StoredUser | None:
    return _USERS.get(username)


def create_access_token(user: StoredUser) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + TOKEN_TTL_SECONDS
    _SESSIONS[token] = TokenRecord(username=user.username, role=user.role, expires_at=expires_at)
    return token, datetime.fromtimestamp(expires_at, timezone.utc).isoformat().replace("+00:00", "Z")


def get_token_from_request(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-API-KEY")


def validate_token(token: str | None) -> User:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    record = _SESSIONS.get(token)
    if record is None or record.expires_at < time.time():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )
    return User(username=record.username, role=record.role)


def auth_required(request: Request) -> User:
    token = get_token_from_request(request)
    return validate_token(token)


def get_current_user(request: Request) -> User:
    return auth_required(request)


def role_required(allowed_roles: list[Role]):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges",
            )
        return user
    return dependency


def login_user(payload: AuthLoginRequest) -> AuthLoginResponse:
    user = _get_user(payload.username)
    if user is None or not _validate_password(payload.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
        )
    token, expires_at = create_access_token(user)
    return AuthLoginResponse(
        access_token=token,
        expires_at=expires_at,
        user=User(username=user.username, role=user.role),
    )
