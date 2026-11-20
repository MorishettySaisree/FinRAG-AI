
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"
USER_DB_PATH = DB_DIR / "users.sqlite3"

load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("FINRAG_SECRET_KEY", "")
ADMIN_USERNAME = os.getenv("FINRAG_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("FINRAG_ADMIN_PASSWORD", "")
ADMIN_EMAIL = os.getenv(
    "FINRAG_ADMIN_EMAIL",
    "admin@example.com",
)

if not SECRET_KEY or len(SECRET_KEY) < 40:
    raise RuntimeError(
        "FINRAG_SECRET_KEY is missing or too short in .env"
    )

if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise RuntimeError(
        "FINRAG_ADMIN_USERNAME and FINRAG_ADMIN_PASSWORD "
        "must be set in .env"
    )

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

password_hasher = PasswordHash.recommended()
bearer_scheme = HTTPBearer(auto_error=False)


# --------------------------------------------------
# Database
# --------------------------------------------------

def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(USER_DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_user_database():
    """Create the account table and seed the existing admin once."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
            """
        )

        existing_admin = connection.execute(
            """
            SELECT user_id
            FROM users
            WHERE user_id = 'admin'
            """
        ).fetchone()

        if existing_admin is None:
            username_conflict = connection.execute(
                """
                SELECT user_id
                FROM users
                WHERE username = ?
                """,
                (ADMIN_USERNAME,),
            ).fetchone()

            if username_conflict is not None:
                raise RuntimeError(
                    "Admin username already belongs to another account."
                )

            connection.execute(
                """
                INSERT INTO users (
                    user_id,
                    full_name,
                    username,
                    email,
                    password_hash,
                    role,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "admin",
                    "FinRAG Administrator",
                    ADMIN_USERNAME,
                    ADMIN_EMAIL,
                    password_hasher.hash(ADMIN_PASSWORD),
                    "admin",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


initialize_user_database()


# --------------------------------------------------
# Account helpers
# --------------------------------------------------

def get_user_by_username(username: str) -> Optional[dict]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT user_id, full_name, username, email,
                   password_hash, role
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT user_id, full_name, username, email, role
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    return dict(row) if row else None


def register_user(
    full_name: str,
    username: str,
    email: str,
    password: str,
) -> dict:
    full_name = full_name.strip()
    username = username.strip()
    email = email.strip().lower()

    if len(full_name) < 2:
        raise ValueError("Enter your full name.")

    if len(username) < 3 or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", username
    ):
        raise ValueError(
            "Username must be at least 3 characters and contain "
            "only letters, numbers, dots, underscores or hyphens."
        )

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("Enter a valid email address.")

    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")

    if username.lower() == ADMIN_USERNAME.lower():
        raise ValueError("This username is reserved.")

    user_id = __import__("uuid").uuid4().hex
    password_hash = password_hasher.hash(password)

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    user_id,
                    full_name,
                    username,
                    email,
                    password_hash,
                    role,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    full_name,
                    username,
                    email,
                    password_hash,
                    "user",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.IntegrityError:
        raise ValueError(
            "That username or email address is already registered."
        )

    return {
        "user_id": user_id,
        "full_name": full_name,
        "username": username,
        "email": email,
        "role": "user",
    }


# --------------------------------------------------
# Login
# --------------------------------------------------

def authenticate_user(
    username: str,
    password: str,
) -> Optional[dict]:
    user = get_user_by_username(username.strip())

    if not user:
        return None

    try:
        password_ok = password_hasher.verify(
            password,
            user["password_hash"],
        )
    except Exception:
        password_ok = False

    if not password_ok:
        return None

    user.pop("password_hash", None)
    return user


# --------------------------------------------------
# JWT
# --------------------------------------------------

def create_access_token(user: dict) -> str:
    now = datetime.now(timezone.utc)

    payload = {
        "sub": user["user_id"],
        "username": user["username"],
        "role": user["role"],
        "iat": now,
        "exp": now + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
) -> str:
    """Return the authenticated account's internal user ID."""

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Login required or token expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]},
        )

        user_id = payload.get("sub")

        if not isinstance(user_id, str):
            raise unauthorized

        user = get_user_by_id(user_id)

        if user is None:
            raise unauthorized

        return user_id

    except jwt.InvalidTokenError:
        raise unauthorized
