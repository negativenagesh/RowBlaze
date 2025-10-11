import httpx
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import os


class Document(BaseModel):
    """Document model for RowBlaze SDK."""

    id: str
    title: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Query result model for RowBlaze SDK."""

    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RowBlazeClient:
    """
    Client for interacting with the RowBlaze API.

    This client provides methods for all core RowBlaze functionalities:
    - Document management (upload, retrieve, delete)
    - Querying documents
    - Chat functionality
    """

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize the RowBlaze client.

        Args:
            api_url: Base URL for the RowBlaze API
            api_key: Optional API key for authentication
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.logger = logging.getLogger("rowblaze.client")

    async def health(self) -> Dict[str, Any]:
        """Check API health status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/health", headers=self.headers, timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    async def upload_document(
        self,
        file_path: str,
        index_name: str = "default",
        description: str = "",
        options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Upload a document to RowBlaze.

        Args:
            file_path: Path to the file to upload
            index_name: Name of the index to add the document to
            description: Optional description of the document
            options: Additional options for processing the document

        Returns:
            Dict containing the upload response
        """
        if options is None:
            options = {}

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"index_name": index_name, "description": description, **options}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/v1/ingest",
                    files=files,
                    data=data,
                    headers=self.headers,
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()

    async def query(
        self,
        question: str,
        index_name: str = "default",
        model: Optional[str] = None,
        options: Dict[str, Any] = None,
    ) -> QueryResult:
        """
        Query the knowledge base.

        Args:
            question: The question to ask
            index_name: Name of the index to query
            model: Optional model override
            options: Additional options for query processing

        Returns:
            QueryResult object containing the answer and sources
        """
        if options is None:
            options = {}

        payload = {
            "question": question,
            "index_name": index_name,
            **({"model": model} if model else {}),
            **options,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/v1/query",
                json=payload,
                headers=self.headers,
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            return QueryResult(**result)

    # Add more methods for other API endpoints...

    # Synchronous wrappers for easier use
    def query_sync(self, *args, **kwargs) -> QueryResult:
        """Synchronous version of query method."""
        return asyncio.run(self.query(*args, **kwargs))

    def upload_document_sync(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous version of upload_document method."""
        return asyncio.run(self.upload_document(*args, **kwargs))

    def health_sync(self) -> Dict[str, Any]:
        """Synchronous version of health method."""
        return asyncio.run(self.health())
