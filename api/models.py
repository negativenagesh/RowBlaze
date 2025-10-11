from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    index_name: str = Field(..., description="Elasticsearch index name")
    file_name: Optional[str] = Field(None, description="Original file name")
    description: Optional[str] = Field(None, description="Document description")
    is_ocr_pdf: bool = Field(False, description="Whether to use OCR for PDF")
    is_structured_pdf: bool = Field(
        False, description="Whether PDF has structured data"
    )
    model: Optional[str] = Field(None, description="LLM model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for LLM")


class IngestionResponse(BaseModel):
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    document_id: Optional[str] = Field(None, description="Generated document ID")
    chunks_count: Optional[int] = Field(None, description="Number of chunks created")


class RetrievalRequest(BaseModel):
    question: str = Field(..., description="User query")
    index_name: str = Field(..., description="Elasticsearch index name")
    top_k_chunks: int = Field(5, description="Number of chunks to retrieve")
    enable_references_citations: bool = Field(
        True, description="Whether to include citations"
    )
    deep_research: bool = Field(
        False, description="Whether to use advanced retrieval techniques"
    )
    auto_chunk_sizing: bool = Field(
        True, description="Whether to auto-determine chunk count"
    )
    model: Optional[str] = Field(None, description="LLM model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for LLM")


class FinalAnswerRequest(BaseModel):
    question: str = Field(..., description="Original user query")
    context: str = Field(..., description="Context with search results")
    generate_final_answer: bool = Field(True, description="Flag to generate answer")
    enable_references_citations: bool = Field(True, description="Include citations")
    model: Optional[str] = Field(None, description="LLM model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for LLM")


class RetrievalResponse(BaseModel):
    answer: str = Field(..., description="Generated answer or search results")
    citations: List[str] = Field(default_factory=list, description="Document citations")
