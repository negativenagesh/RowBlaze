import logging
from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Depends, HTTPException, status
from openai import AsyncOpenAI

from api.dependencies import get_elasticsearch_client, get_openai_client
from api.models import FinalAnswerRequest, RetrievalRequest, RetrievalResponse
from sdk.message import Message
from sdk.response import FunctionResponse, Messages
from src.core.retrieval.rag_retrieval import RAGFusionRetriever

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=RetrievalResponse)
async def query_documents(
    request: RetrievalRequest,
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
):
    """
    Search for information and generate answers based on indexed documents.
    """
    logger.info(f"Query request received: {request.question}")

    try:
        # Create retriever
        params = request.dict()
        config = {"index_name": request.index_name}

        retriever = RAGFusionRetriever(params, config, es_client, openai_client)

        # Determine optimal chunk count if enabled
        if request.auto_chunk_sizing:
            try:
                top_k_chunks = await retriever._determine_optimal_chunk_count(
                    request.question
                )
                logger.info(f"Using dynamically determined chunk count: {top_k_chunks}")
            except Exception as e:
                logger.warning(
                    f"Error determining optimal chunk count: {e}. Using provided value."
                )
                top_k_chunks = request.top_k_chunks
        else:
            top_k_chunks = request.top_k_chunks

        # Perform search
        search_results = await retriever.search(
            user_query=request.question,
            initial_candidate_pool_size=top_k_chunks,
            top_k_kg_entities=top_k_chunks,
            absolute_score_floor=0.3,
        )

        # Generate final answer if needed
        if not params.get("skip_final_answer", False):
            llm_formatted_context = search_results.get("llm_formatted_context", "")

            # Extract citations
            cited_files = []
            if "refrences" in search_results:
                refs_text = search_results["refrences"]
                if refs_text and "**Sources:**" in refs_text:
                    sources_section = refs_text.split("**Sources:**")[1].split("\n\n")[
                        0
                    ]
                    cited_files = [
                        line.replace("- ", "").strip()
                        for line in sources_section.strip().split("\n")
                    ]

            # Generate answer
            final_answer = await retriever._generate_final_answer(
                original_query=request.question,
                llm_formatted_context=llm_formatted_context,
                cited_files=cited_files,
                model=request.model,
            )

            return RetrievalResponse(answer=final_answer, citations=cited_files)

        # Return formatted context if no final answer
        return RetrievalResponse(
            answer=search_results.get(
                "llm_formatted_context", "No formatted context generated."
            ),
            citations=[],
        )

    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during retrieval: {str(e)}",
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
