import logging
from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Depends, HTTPException, status
from openai import AsyncOpenAI

from api.dependencies import get_elasticsearch_client, get_openai_client
from api.models import FinalAnswerRequest, RetrievalRequest, RetrievalResponse
from sdk.message import Message
from sdk.response import FunctionResponse, Messages
from src.core.agents.static_research_agent import StaticResearchAgent  # Add this import
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
    Search for information and generate answers based on indexed documents using the Static Research Agent.
    """
    logger.info(f"Query request received: {request.question}")

    try:
        # Create RAGFusionRetriever for the agent
        params = request.dict()
        config = {"index_name": request.index_name}
        retriever = RAGFusionRetriever(params, config, es_client, openai_client)

        # Create and configure the Static Research Agent
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=retriever,
            llm_model=request.model,
            max_iterations=5,  # You can make this configurable
        )

        # Create agent config override with request parameters
        agent_config_override = {
            "temperature": 0.3,
            "max_tokens_llm_response": request.max_tokens,
            "initial_retrieval_subqueries": 2,
            "initial_retrieval_top_k_chunks": request.top_k_chunks,
            "initial_retrieval_top_k_kg": request.top_k_chunks,
            "perform_initial_retrieval": True,
            "current_date": "today",
        }

        # Run the agent with the user's query
        logger.info(f"Running Static Research Agent for query: {request.question}")
        agent_result = await agent.arun(
            query=request.question, agent_config_override=agent_config_override
        )

        # Extract the answer and any available citations
        final_answer = agent_result.get("answer", "No answer generated")

        # Extract citations from tools used (if any search tools were used)
        cited_files = []
        tools_used = agent_result.get("tools_used", [])

        # You can extract file references from the conversation history
        # or implement a method to track citations through the agent's tool usage
        for tool_call in tools_used:
            if tool_call.get("tool_name") in [
                "search_file_knowledge",
                "vector_search",
                "keyword_search",
            ]:
                # You might want to implement citation tracking in the agent's tools
                pass

        logger.info(f"Agent completed with {len(tools_used)} tools used")

        return RetrievalResponse(
            answer=final_answer, citations=cited_files
        )  # Return citations if available

    except Exception as e:
        logger.error(f"Error during agent processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during agent processing: {str(e)}",
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
