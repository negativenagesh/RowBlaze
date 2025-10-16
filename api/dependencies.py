# api/dependencies.py

import os
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Optional

import jwt
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Initialize constants
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"


@lru_cache()
def get_settings():
    return {
        "elasticsearch_url": os.getenv("RAG_UPLOAD_ELASTIC_URL"),
        "elasticsearch_api_key": os.getenv("ELASTICSEARCH_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "openai_embedding_model": os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
        ),
    }


async def get_elasticsearch_client() -> Optional[AsyncElasticsearch]:
    """
    Initializes and returns an AsyncElasticsearch client.
    Handles None if the connection fails.
    """
    # CORRECTED: Load environment variables directly into the function's scope.
    # This fixes the "name 'ELASTICSEARCH_URL' is not defined" error.
    ELASTICSEARCH_URL = os.getenv("RAG_UPLOAD_ELASTIC_URL")
    ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

    if not ELASTICSEARCH_URL:
        print("❌ ELASTICSEARCH_URL is not set in the environment.")
        return None

    try:
        es_client = None
        if ELASTICSEARCH_API_KEY:
            # CORRECTED: The client expects the URL in a list via the `hosts` parameter.
            es_client = AsyncElasticsearch(
                hosts=[ELASTICSEARCH_URL],
                api_key=ELASTICSEARCH_API_KEY,
                request_timeout=60,
                retry_on_timeout=True,
            )
        else:
            es_client = AsyncElasticsearch(
                hosts=[ELASTICSEARCH_URL], request_timeout=60, retry_on_timeout=True
            )

        if not await es_client.ping():
            print("❌ Ping to Elasticsearch cluster failed.")
            return None

        print("✅ AsyncElasticsearch client initialized.")
        return es_client
    except Exception as e:
        print(f"❌ Failed to initialize AsyncElasticsearch client: {e}")
        return None


async def get_openai_client() -> AsyncOpenAI:
    """Get AsyncOpenAI client as a FastAPI dependency."""
    settings = get_settings()

    if not settings["openai_api_key"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key is not configured",
        )

    try:
        client = AsyncOpenAI(api_key=settings["openai_api_key"])
        return client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize OpenAI: {str(e)}",
        )


# Function to get request parameters
def get_params():
    """Get parameters from request."""
    return {}


# Token verification function
async def verify_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Verify JWT token and return payload."""
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret is not configured",
        )

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Current user dependency
async def get_current_user(
    payload: Dict[str, Any] = Depends(verify_token),
) -> Dict[str, Any]:
    """Extract current user from token payload."""
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": user_id,
        "username": payload.get("username"),
        "email": payload.get("email"),
    }
