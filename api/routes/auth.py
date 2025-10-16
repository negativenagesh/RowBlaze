import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, validator

# Import get_current_user from dependencies
from api.dependencies import get_current_user

router = APIRouter()

# --- Configuration ---
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days (like Claude)


# --- Pydantic Models ---
class UserRegistration(BaseModel):
    email: str
    password: str

    @validator("email")
    def validate_email_format(cls, v):
        if not validate_email(v):
            raise ValueError("Invalid email format")
        return v.lower().strip()

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        return v


class UserLogin(BaseModel):
    email: str
    password: str

    @validator("email")
    def validate_email_format(cls, v):
        if not validate_email(v):
            raise ValueError("Invalid email format")
        return v.lower().strip()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


# --- In-Memory User Database (Replace with real database in production) ---
USERS_DB: Dict[str, Dict[str, Any]] = {}


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using bcrypt."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    if not email or len(email) > 254:
        return False

    # Basic email regex pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    # Additional checks
    if not re.match(pattern, email):
        return False

    # Check for consecutive dots
    if ".." in email:
        return False

    # Check local part length (before @)
    local_part = email.split("@")[0]
    if len(local_part) > 64:
        return False

    return True


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    # JWT expects timestamp, not ISO string
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


@router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin):
    """
    Authenticate user with email and password.
    """
    # Find user by email
    user = None
    for user_data in USERS_DB.values():
        if user_data.get("email") == login_data.email:
            user = user_data
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled",
        )

    # Update last login
    user["last_login"] = datetime.now(timezone.utc).isoformat()

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user["id"],
            "email": user["email"],
            "iat": datetime.now(timezone.utc).timestamp(),
        },
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=access_token,
        user={
            "id": user["id"],
            "email": user["email"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
        },
    )


# Legacy endpoint for OAuth2 compatibility
@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2-compatible token endpoint (uses email as username).
    """
    login_data = UserLogin(email=form_data.username, password=form_data.password)
    result = await login_user(login_data)
    return {"access_token": result.access_token, "token_type": "bearer"}


@router.post("/oauth/github")
async def github_oauth_login(github_data: Dict[str, Any]):
    """
    Handle GitHub OAuth login/registration.
    """
    try:
        github_id = github_data.get("github_id")
        email = github_data.get("email")
        name = github_data.get("name", "")
        avatar_url = github_data.get("avatar_url", "")

        if not github_id or not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required GitHub user data",
            )

        # Check if user already exists
        existing_user = USERS_DB.get(email)
        if existing_user:
            # Update last login
            existing_user["last_login"] = datetime.now(timezone.utc).isoformat()
            user_id = existing_user["id"]
        else:
            # Create new user
            user_id = f"github_{github_id}"
            current_time = datetime.now(timezone.utc).isoformat()
            new_user = {
                "id": user_id,
                "email": email,
                "password_hash": "",  # GitHub users don't have passwords
                "name": name,
                "avatar_url": avatar_url,
                "github_id": github_id,
                "disabled": False,
                "created_at": current_time,
                "last_login": current_time,
                "email_verified": True,  # GitHub emails are verified
                "auth_provider": "github",
            }
            USERS_DB[email] = new_user

        # Generate access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user_id,
                "email": email,
                "iat": datetime.now(timezone.utc).timestamp(),
            },
            expires_delta=access_token_expires,
        )

        return TokenResponse(
            access_token=access_token,
            user={
                "id": user_id,
                "email": email,
                "name": name,
                "avatar_url": avatar_url,
                "created_at": USERS_DB[email]["created_at"],
                "last_login": USERS_DB[email]["last_login"],
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub OAuth error: {str(e)}",
        )


@router.post("/register", response_model=TokenResponse)
async def register_user(registration_data: UserRegistration):
    """
    Register a new user with email and password.
    """
    # Validate email format
    if not validate_email(registration_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email format"
        )

    # Check if email already exists
    for user_data in USERS_DB.values():
        if user_data.get("email") == registration_data.email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists",
            )

    # Generate user ID
    user_id = str(uuid.uuid4())

    # Hash password
    password_hash = hash_password(registration_data.password)

    # Create new user
    current_time = datetime.now(timezone.utc).isoformat()
    new_user = {
        "id": user_id,
        "email": registration_data.email,
        "password_hash": password_hash,
        "disabled": False,
        "created_at": current_time,
        "last_login": None,
        "email_verified": False,  # In production, implement email verification
    }

    # Add to database (using email as key for easy lookup)
    USERS_DB[registration_data.email] = new_user

    # Generate access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_id,
            "email": registration_data.email,
            "iat": datetime.now(timezone.utc).timestamp(),
        },
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=access_token,
        user={
            "id": user_id,
            "email": registration_data.email,
            "created_at": current_time,
            "last_login": None,
        },
    )


@router.get("/me")
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current user information."""
    user_email = current_user.get("email")
    if user_email and user_email in USERS_DB:
        user_data = USERS_DB[user_email]
        return {
            "id": user_data["id"],
            "email": user_data["email"],
            "created_at": user_data["created_at"],
            "last_login": user_data.get("last_login"),
            "email_verified": user_data.get("email_verified", False),
        }

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


@router.post("/logout")
async def logout_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout user (client should discard the token)."""
    return {"message": "Successfully logged out"}


# Auth health endpoint removed - using main health endpoint instead


# Debug endpoint (remove in production)
@router.get("/debug/users")
async def debug_users():
    """Debug endpoint to check user database state."""
    if os.getenv("ENVIRONMENT") == "development":
        return {
            "total_users": len(USERS_DB),
            "emails": list(USERS_DB.keys()),
            "users": {k: {**v, "password_hash": "***"} for k, v in USERS_DB.items()},
        }
    else:
        raise HTTPException(status_code=404, detail="Not found")
