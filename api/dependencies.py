import os
from functools import lru_cache
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from fastapi import Depends, HTTPException, status
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


@lru_cache()
def get_settings():
    return {
        "elasticsearch_url": os.getenv("RAG_UPLOAD_ELASTIC_URL"),
        "elasticsearch_api_key": os.getenv("ELASTICSEARCH_API_KEY"),
        "openai_api_key": os.getenv("OPEN_AI_KEY"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "openai_embedding_model": os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
        ),
    }


async def get_elasticsearch_client() -> AsyncGenerator[AsyncElasticsearch, None]:
    """Get AsyncElasticsearch client as a FastAPI dependency."""
    settings = get_settings()

    try:
        if settings["elasticsearch_api_key"]:
            client = AsyncElasticsearch(
                settings["elasticsearch_url"],
                api_key=settings["elasticsearch_api_key"],
                request_timeout=60,
                retry_on_timeout=True,
            )
        else:
            client = AsyncElasticsearch(
                hosts=[settings["elasticsearch_url"]],
                request_timeout=60,
                retry_on_timeout=True,
            )

        if not await client.ping():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch connection failed",
            )

        try:
            yield client
        finally:
            await client.close()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize Elasticsearch: {str(e)}",
        )


async def get_openai_client() -> AsyncGenerator[AsyncOpenAI, None]:
    """Get AsyncOpenAI client as a FastAPI dependency."""
    settings = get_settings()

    if not settings["openai_api_key"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key is not configured",
        )

    try:
        client = AsyncOpenAI(api_key=settings["openai_api_key"])
        yield client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize OpenAI: {str(e)}",
        )
