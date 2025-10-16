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


# NEW: Add missing QueryRequest model
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question to answer")
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


# NEW: Add missing QueryResponse model
class QueryResponse(BaseModel):
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer")
    context: List[Any] = Field(
        default_factory=list, description="Retrieved context chunks"
    )
    cited_files: List[str] = Field(
        default_factory=list, description="Files cited in answer"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# NEW: Add hierarchy models
class HierarchyLevel(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    # Note: description is not in CHUNKED_PDF_MAPPINGS but may be present in data
    description: Optional[str] = None


class Hierarchy(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    root_type: Optional[str] = None
    levels: List[HierarchyLevel] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)


class SearchResult(BaseModel):
    text: str
    score: float
    file_name: str
    page_number: int
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    hierarchies: List[Hierarchy] = Field(default_factory=list)  # NEW: Add hierarchies


class RetrievalResponse(BaseModel):
    answer: str
    context: str
    search_results: List[SearchResult]
    query_type: Optional[str] = None
    confidence_score: Optional[float] = None
