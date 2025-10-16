import logging
import os
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Depends, HTTPException, status
from openai import AsyncOpenAI

from api.dependencies import get_elasticsearch_client, get_openai_client
from api.models import (
    FinalAnswerRequest,
    QueryRequest,
    QueryResponse,
    RetrievalRequest,
    RetrievalResponse,
)

from ..middleware.auth_middleware import get_current_user

# Import the retrieval system components
try:
    from src.core.retrieval.rag_retrieval import RAGFusionRetriever
except ImportError:
    RAGFusionRetriever = None

try:
    from src.core.agents.static_research_agent import StaticResearchAgent
except ImportError:
    StaticResearchAgent = None

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def retrieval_health():
    """Health check for retrieval service."""
    return {
        "status": "healthy",
        "service": "retrieval",
        "rag_available": RAGFusionRetriever is not None,
        "agent_available": StaticResearchAgent is not None,
    }


@router.post("/query", response_model=RetrievalResponse)
async def query_documents(
    request: RetrievalRequest,
    current_user=Depends(get_current_user),
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
):
    """
    Search for information and generate answers based on indexed documents using the Static Research Agent.
    """
    logger.info(f"Query request received: {request.question}")

    if not es_client:
        raise HTTPException(
            status_code=503, detail="Elasticsearch client not available"
        )
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not available")

    try:
        # User-specific index name
        user_id = current_user.get("user_id") or current_user.get("sub", "default")
        index_name = f"rowblaze-{user_id}"

        # Create parameters for the retrieval system
        params = {
            "question": request.question,
            "top_k_chunks": getattr(request, "top_k_chunks", 5),
            "enable_references_citations": getattr(
                request, "enable_references_citations", True
            ),
            "deep_research": getattr(request, "deep_research", False),
            "auto_chunk_sizing": getattr(request, "auto_chunk_sizing", True),
            "model": getattr(request, "model", None)
            or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        }

        config = {"index_name": index_name}

        # Check if components are available
        if not RAGFusionRetriever:
            # Fallback to simple response
            return RetrievalResponse(
                answer="RAG system not available. Please check system configuration.",
                context="System configuration error",
                search_results=[],
                query_type="error",
                confidence_score=0.0,
            )

        # Create retriever
        logger.info(f"Creating RAGFusionRetriever with params: {params}")
        retriever = RAGFusionRetriever(params, config, es_client, openai_client)

        if StaticResearchAgent:
            # Use the research agent if available
            logger.info(f"Creating StaticResearchAgent with params: {params}")
            agent = StaticResearchAgent(
                llm_client=openai_client,
                retriever=retriever,
                params=params,
            )

            # Configure agent
            agent_config = {
                "initial_retrieval_subqueries": getattr(request, "num_subqueries", 2),
                "initial_retrieval_top_k_chunks": getattr(request, "top_k_chunks", 20),
                "initial_retrieval_top_k_kg": getattr(request, "top_k_kg", 10),
            }

            # Run the agent
            result = await agent.arun(request.question, agent_config)

            # Extract results
            answer = result.get("answer", "No answer generated.")
            context = result.get("context", "No context available.")

            # Format search results
            search_results = []
            iteration_results = result.get("iteration_results", {})
            for iteration_data in iteration_results.values():
                if (
                    isinstance(iteration_data, dict)
                    and "search_results" in iteration_data
                ):
                    for sr in iteration_data["search_results"]:
                        search_results.append(
                            {
                                "text": sr.get("text", ""),
                                "score": sr.get("score", 0.0),
                                "file_name": sr.get("file_name", ""),
                                "page_number": sr.get("page_number", 1),
                                "entities": sr.get("entities", []),
                                "relationships": sr.get("relationships", []),
                            }
                        )

            return RetrievalResponse(
                answer=answer,
                context=context,
                search_results=search_results,
                query_type=result.get("query_type", "unknown"),
                confidence_score=0.8,
            )
        else:
            # Fallback to direct retriever usage
            results = await retriever.process_query(request.question)

            return RetrievalResponse(
                answer=results.get("answer", "No answer generated."),
                context=results.get("context", "No context available."),
                search_results=results.get("search_results", []),
                query_type="direct_retrieval",
                confidence_score=0.7,
            )

    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


@router.post("/generate-answer", response_model=RetrievalResponse)
async def generate_final_answer(
    request: FinalAnswerRequest, openai_client: AsyncOpenAI = Depends(get_openai_client)
):
    """
    Generate a final answer from provided context.
    """
    logger.info(f"Final answer generation requested for query: {request.question}")

    try:
        # Create retriever (without ES client for answer generation only)
        params = request.dict()
        config = {}
        retriever = RAGFusionRetriever(params, config, None, openai_client)

        # Extract cited files from context if available
        cited_files = []
        if "**Sources:**" in request.context:
            sources_section = request.context.split("**Sources:**")[1].split("\n\n")[0]
            cited_files = [
                line.replace("- ", "").strip()
                for line in sources_section.strip().split("\n")
            ]

        # Generate the answer
        final_answer = await retriever._generate_final_answer(
            original_query=request.question,
            llm_formatted_context=request.context,
            cited_files=cited_files,
            model=request.model,
        )

        return RetrievalResponse(answer=final_answer, citations=cited_files)

    except Exception as e:
        logger.error(f"Error generating final answer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating final answer: {str(e)}",
        )


async def handle_request(
    params: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle retrieval requests using the RAGFusionRetriever.
    """
    es_client = await get_elasticsearch_client()
    openai_client = await get_openai_client()

    if not es_client or not openai_client:
        return {
            "answer": "System unavailable",
            "context": [],
            "cited_files": [],
            "metadata": {"error": "Database or AI service unavailable"},
        }

    try:
        if RAGFusionRetriever:
            # Create retriever with the ES and OpenAI clients
            retriever = RAGFusionRetriever(params, config, es_client, openai_client)

            # Use the search method (process_query doesn't exist)
            results = await retriever.search(
                user_query=params["question"],
                num_subqueries=params.get("num_subqueries", 2),
                initial_candidate_pool_size=params.get("top_k_chunks", 20),
                top_k_kg_entities=params.get("top_k_kg", 10),
            )

            return {
                "answer": results.get("answer", ""),
                "context": results.get("context", []),
                "cited_files": results.get("cited_files", []),
                "metadata": results.get("metadata", {}),
            }
        else:
            return {
                "answer": "RAG system not available",
                "context": [],
                "cited_files": [],
                "metadata": {"error": "RAG system not configured"},
            }
    except Exception as e:
        logger.error(f"Error in handle_request: {str(e)}", exc_info=True)
        return {
            "answer": f"Error processing request: {str(e)}",
            "context": [],
            "cited_files": [],
            "metadata": {"error": str(e)},
        }


@router.get("/chunks/{index_name}")
async def get_chunks_by_file(
    index_name: str,
    file_name: str = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
):
    """Get chunks for a specific file or all chunks in the index."""
    try:
        # Check if index exists
        if not await es_client.indices.exists(index=index_name):
            logger.warning(f"Index {index_name} does not exist")
            return {"success": True, "chunks": [], "total": 0}

        query = {"match_all": {}}
        if file_name:
            # Try different field variations for file name matching
            query = {
                "bool": {
                    "should": [
                        {"term": {"metadata.file_name.keyword": file_name}},
                        {"term": {"metadata.file_name": file_name}},
                        {"match": {"metadata.file_name": file_name}},
                        {"wildcard": {"metadata.file_name": f"*{file_name}*"}},
                    ],
                    "minimum_should_match": 1,
                }
            }

        # Remove problematic sort fields and use simpler sorting
        response = await es_client.search(
            index=index_name,
            query=query,
            size=1000,
            _source_includes=["chunk_text", "metadata"],
        )

        chunks = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            chunks.append(
                {
                    "id": hit.get("_id"),
                    "text": source.get("chunk_text", ""),
                    "file_name": metadata.get("file_name", ""),
                    "page_number": metadata.get("page_number", 1),
                    "chunk_index": metadata.get("chunk_index_in_page", 0),
                    "document_summary": metadata.get("document_summary", ""),
                }
            )

        # Sort chunks by file name and page number in Python since ES sort failed
        chunks.sort(
            key=lambda x: (
                x.get("file_name", ""),
                x.get("page_number", 0),
                x.get("chunk_index", 0),
            )
        )

        return {"success": True, "chunks": chunks, "total": len(chunks)}

    except Exception as e:
        logger.error(f"Error fetching chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching chunks: {str(e)}",
        )


@router.get("/knowledge-graph/{index_name}")
async def get_knowledge_graph_by_file(
    index_name: str,
    file_name: str = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
):
    """Get knowledge graph data for a specific file or all files in the index."""
    try:
        # Check if index exists
        if not await es_client.indices.exists(index=index_name):
            logger.warning(f"Index {index_name} does not exist")
            return {
                "success": True,
                "knowledge_graph": {
                    "entities": [],
                    "relationships": [],
                    "hierarchies": [],
                },
                "total_entities": 0,
                "total_relationships": 0,
                "total_hierarchies": 0,
            }

        query = {"match_all": {}}
        if file_name:
            # Try different field variations for file name matching
            query = {
                "bool": {
                    "should": [
                        {"term": {"metadata.file_name.keyword": file_name}},
                        {"term": {"metadata.file_name": file_name}},
                        {"match": {"metadata.file_name": file_name}},
                        {"wildcard": {"metadata.file_name": f"*{file_name}*"}},
                    ],
                    "minimum_should_match": 1,
                }
            }

        # Remove problematic sort fields
        response = await es_client.search(
            index=index_name,
            query=query,
            size=1000,
            _source_includes=[
                "metadata.entities",
                "metadata.relationships",
                "metadata.hierarchies",
                "metadata.file_name",
                "metadata.page_number",
            ],
        )

        entities = []
        relationships = []
        hierarchies = []

        for hit in response.get("hits", {}).get("hits", []):
            metadata = hit.get("_source", {}).get("metadata", {})
            file_name_meta = metadata.get("file_name", "")
            page_number = metadata.get("page_number", 1)

            # Collect entities
            for entity in metadata.get("entities", []):
                entities.append(
                    {**entity, "file_name": file_name_meta, "page_number": page_number}
                )

            # Collect relationships
            for relationship in metadata.get("relationships", []):
                relationships.append(
                    {
                        **relationship,
                        "file_name": file_name_meta,
                        "page_number": page_number,
                    }
                )

            # Collect hierarchies - FIXED: Properly handle the nested structure according to CHUNKED_PDF_MAPPINGS
            for hierarchy in metadata.get("hierarchies", []):
                # Ensure hierarchy has the correct structure according to CHUNKED_PDF_MAPPINGS
                hierarchy_item = {
                    "name": hierarchy.get("name", "Unnamed Hierarchy"),
                    "description": hierarchy.get("description", ""),
                    "root_type": hierarchy.get("root_type", "Unknown"),
                    "levels": [],
                    "relationships": hierarchy.get("relationships", []),
                    "file_name": file_name_meta,
                    "page_number": page_number,
                }

                # Process levels properly - levels is nested with id, name, and nodes (per CHUNKED_PDF_MAPPINGS)
                for level in hierarchy.get("levels", []):
                    level_item = {
                        "id": level.get("id", ""),
                        "name": level.get("name", ""),
                        "nodes": level.get(
                            "nodes", []
                        ),  # nodes is nested type in mapping
                    }
                    # Only add description if it exists in the source data (not in mapping but may be present)
                    if level.get("description"):
                        level_item["description"] = level.get("description", "")
                    hierarchy_item["levels"].append(level_item)

                hierarchies.append(hierarchy_item)

        # Sort entities and relationships by file name and page number in Python
        entities.sort(key=lambda x: (x.get("file_name", ""), x.get("page_number", 0)))
        relationships.sort(
            key=lambda x: (x.get("file_name", ""), x.get("page_number", 0))
        )
        hierarchies.sort(
            key=lambda x: (x.get("file_name", ""), x.get("page_number", 0))
        )

        return {
            "success": True,
            "knowledge_graph": {
                "entities": entities,
                "relationships": relationships,
                "hierarchies": hierarchies,
            },
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "total_hierarchies": len(hierarchies),
        }

    except Exception as e:
        logger.error(f"Error fetching knowledge graph: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching knowledge graph: {str(e)}",
        )


@router.post("/query-rag", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
) -> QueryResponse:
    """Process a query against the RAG system."""
    try:
        logger.info(f"Query request received: {request.question}")

        # User-specific index name
        user_id = current_user.get("user_id") or current_user.get("sub", "default")
        index_name = f"rowblaze-{user_id}"

        # Create parameters for the retrieval system
        params = {
            "question": request.question,
            "top_k_chunks": request.top_k_chunks,
            "enable_references_citations": request.enable_references_citations,
            "deep_research": request.deep_research,
            "auto_chunk_sizing": request.auto_chunk_sizing,
            "model": request.model or "gpt-4o-mini",
        }

        config = {"index_name": index_name}

        # Call the retrieval handler
        response = await handle_request(params, config)

        # Return the response
        return QueryResponse(
            question=request.question,
            answer=response.get("answer", "Failed to generate answer."),
            context=response.get("context", []),
            cited_files=response.get("cited_files", []),
            metadata=response.get("metadata", {}),
        )
    except Exception as e:
        logger.error(f"Error in query processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"Query processing failed: {str(e)}"
        )


@router.post("/agent-query", response_model=QueryResponse)
async def agent_query_rag(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
) -> QueryResponse:
    """Process a query using the Agentic RAG system with StaticResearchAgent for COMPLETE retrieval."""
    agent = None
    try:
        logger.info(f"🧠 Agentic RAG query request received: {request.question}")

        # User-specific index name
        user_id = current_user.get("user_id") or current_user.get("sub", "default")
        index_name = f"rowblaze-{user_id}"

        # Import the StaticResearchAgent
        try:
            from src.core.agents.static_research_agent import StaticResearchAgent
            from src.core.retrieval.rag_retrieval import RAGFusionRetriever
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise HTTPException(
                status_code=500,
                detail="Agentic RAG system not available. Please check system configuration.",
            )

        # Create parameters for the agent - COMPLETE AGENTIC RETRIEVAL
        agent_params = {
            "question": request.question,
            "top_k_chunks": request.top_k_chunks,
            "enable_references_citations": request.enable_references_citations,
            "deep_research": request.deep_research,
            "auto_chunk_sizing": request.auto_chunk_sizing,
            "model": request.model or "gpt-4o-mini",
            "max_tokens": getattr(request, "max_tokens", 16384),
            "max_iterations": getattr(request, "max_iterations", 2),
            "index_name": index_name,
        }

        # Agent config for COMPLETE agentic behavior - NO initial retrieval, let agent handle everything
        agent_config = {
            "index_name": index_name,
            "perform_initial_retrieval": False,  # CRITICAL: Let agent handle ALL retrieval via tools
            "max_iterations": 2,
            "temperature": 0.3,
            "max_tokens_llm_response": 16000,
            # Tool-specific configurations
            "tool_top_k_chunks": request.top_k_chunks,
            "tool_top_k_kg": 10,
            "tool_num_subqueries": 2,
        }

        # Create retriever for the agent (used by tools, not for initial retrieval)
        retriever = RAGFusionRetriever(
            params=agent_params,
            config=agent_config,
            es_client=es_client,
            aclient_openai=openai_client,
        )

        # Create the agent with NO initial retrieval - pure agentic mode
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=retriever,  # Tools will use this
            llm_model=request.model or "gpt-4o-mini",
            max_iterations=2,
            params=agent_params,
        )

        logger.info(
            f"🔧 Agentic mode: Agent initialized with {len(agent.tools)} tools: {list(agent.tools.keys())}"
        )
        logger.info(
            "🎯 Pure agentic mode: No initial retrieval, agent will use tools for ALL information gathering"
        )

        # Run the agent in PURE agentic mode - it will use tools for everything
        agent_result = await agent.arun(request.question, agent_config)

        logger.info(
            f"✅ Agentic agent completed with {agent_result.get('iterations_completed', 0)} iterations"
        )

        # Extract results from agent
        answer = agent_result.get("answer", "No answer generated by agent.")

        # Format metadata with agentic-specific information
        metadata = {
            "query_type": agent_result.get("query_type", "unknown"),
            "dynamic_top_k": agent_result.get("dynamic_top_k", request.top_k_chunks),
            "iterations_completed": agent_result.get("iterations_completed", 0),
            "tools_used": agent_result.get("tools_used", []),
            "agent_mode": "pure_agentic_rag",  # Indicate this is pure agentic mode
            "sufficiency_assessment": agent_result.get("sufficiency_assessment", {}),
            "iteration_results": agent_result.get("iteration_results", {}),
            "skipped_iterations": agent_result.get("skipped_iterations", False),
            "initial_retrieval_performed": False,  # No initial retrieval in pure agentic mode
        }

        # Extract context and citations from agent tool results
        context = []
        cited_files = []

        # Extract from iteration results (tool outputs)
        iteration_results = agent_result.get("iteration_results", {})
        for iteration_key, iteration_data in iteration_results.items():
            if isinstance(iteration_data, str):
                # Look for file references in tool outputs
                import re

                file_matches = re.findall(r"File: ([^,\n\]]+)", iteration_data)
                cited_files.extend(file_matches)

                # Add iteration data as context
                if "Tool:" in iteration_data:
                    context.append(iteration_data)

        # Extract from tools_used metadata
        tools_used = agent_result.get("tools_used", [])
        for tool_info in tools_used:
            tool_name = tool_info.get("tool_name", "")
            if tool_name:
                context.append(
                    f"Used tool: {tool_name} with parameters: {tool_info.get('parameters', {})}"
                )

        # Remove duplicates from cited files
        cited_files = list(set([f.strip() for f in cited_files if f.strip()]))

        logger.info(
            f"🎯 Agentic retrieval complete: {len(tools_used)} tools used, {len(cited_files)} files cited"
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            context=context,
            cited_files=cited_files,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Error in agentic query processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"Agentic query processing failed: {str(e)}"
        )
    finally:
        # Cleanup agent resources
        if agent:
            try:
                await agent.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Error during agent cleanup: {cleanup_error}")
