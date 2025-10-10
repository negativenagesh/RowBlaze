import os
import tempfile
import hashlib
import logging
from typing import List
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI

from api.models import IngestionRequest, IngestionResponse
from api.dependencies import get_elasticsearch_client, get_openai_client
from src.core.ingestion.rag_ingestion import example_run_file_processing
from sdk.message import Message

logger = logging.getLogger(__name__)
router = APIRouter()

def _generate_doc_id_from_content(content_bytes: bytes) -> str:
    """Generates a SHA256 hash for the given byte content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()

@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_name: str = Form(...),
    description: str = Form(None),
    is_ocr_pdf: bool = Form(False),
    is_structured_pdf: bool = Form(False),
    model: str = Form(None),
    max_tokens: int = Form(None),
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
    openai_client: AsyncOpenAI = Depends(get_openai_client)
):
    """
    Ingest a document into the Elasticsearch index.
    """
    logger.info(f"Ingestion request received for file: {file.filename}, index: {index_name}")
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp:
        content = await file.read()
        temp.write(content)
        temp_path = temp.name
    
    try:
        # Generate document ID from content
        document_id = _generate_doc_id_from_content(content)
        logger.info(f"Generated document ID: {document_id}")
        
        # Create parameters for ingestion
        params = {
            "index_name": index_name,
            "file_name": file.filename,
            "file_path": temp_path,
            "description": description or f"Uploaded document: {file.filename}",
            "is_ocr_pdf": is_ocr_pdf,
            "is_structured_pdf": is_structured_pdf,
        }
        
        if model:
            params["model"] = model
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        config = {
            "api_key": os.getenv("OPEN_AI_KEY"),
        }
        
        # Process document in the background
        background_tasks.add_task(
            example_run_file_processing,
            file_data=content,
            original_file_name=file.filename,
            document_id=document_id,
            user_provided_doc_summary=params["description"],
            es_client=es_client,
            aclient_openai=openai_client,
            params=params,
            config=config
        )
        
        return IngestionResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            document_id=document_id
        )
    
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"❌ Error processing document: {str(e)}"  # Keep error symbol for actual errors
        )
    finally:
        # Ensure temp file is cleaned up
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass

@router.get("/files/{index_name}", response_model=List[str])
async def list_files(
    index_name: str,
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client)
):
    """
    List all unique files in the specified index.
    """
    try:
        # Check if index exists
        if not await es_client.indices.exists(index=index_name):
            return []
        
        # Get aggregation of unique file names
        # CHANGE: Use metadata.file_name.keyword instead of metadata.file_name
        response = await es_client.search(
            index=index_name,
            body={
                "size": 0,
                "aggs": {
                    "unique_files": {
                        "terms": {
                            "field": "metadata.file_name.keyword",  # Changed .keyword suffix
                            "size": 1000
                        }
                    }
                }
            }
        )
        
        # Extract file names from the aggregation buckets
        unique_files = []
        if "aggregations" in response and "unique_files" in response["aggregations"]:
            buckets = response["aggregations"]["unique_files"]["buckets"]
            for bucket in buckets:
                file_name = bucket["key"]
                unique_files.append(file_name)
        
        return sorted(unique_files)
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error listing files: {str(e)}"
        )

@router.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint for the API.
    """
    return {"status": "ok", "message": "API is running"}