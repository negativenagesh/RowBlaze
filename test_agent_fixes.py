#!/usr/bin/env python3
"""
Test script to verify the StaticResearchAgent fixes work correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.core.agents.static_research_agent import StaticResearchAgent
from src.core.retrieval.rag_retrieval import RAGFusionRetriever

load_dotenv()


async def test_agent_sufficiency_logic():
    """Test that the agent correctly handles sufficient initial results."""

    # Mock OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "test-key"))

    # Create a mock retriever that returns good results
    class MockRetriever:
        async def search(
            self,
            user_query,
            num_subqueries=2,
            initial_candidate_pool_size=20,
            top_k_kg_entities=10,
        ):
            return {
                "llm_formatted_context": f"""
                Original Query: {user_query}

                --- Results for Sub-query: "{user_query}" ---

                Vector Search Results (Chunks):
                Source ID [c_abc123_p1_i1]: (Score: 0.85)
                This document contains comprehensive information about {user_query}.
                The answer to your question is clearly stated here with detailed explanations
                and supporting evidence. All relevant aspects are covered thoroughly.
                File: test_document.pdf, Page: 1, Chunk Index in Page: 1

                Source ID [c_def456_p2_i1]: (Score: 0.78)
                Additional supporting information that reinforces the main answer.
                This provides context and background that makes the response complete.
                File: supporting_doc.pdf, Page: 2, Chunk Index in Page: 1
                """,
                "sub_queries_results": [],
            }

    # Create agent with mock retriever
    agent = StaticResearchAgent(
        llm_client=openai_client,
        retriever=MockRetriever(),
        params={"model": "gpt-4o-mini"},
    )

    # Test query
    test_query = "What is the main topic discussed in the documents?"

    print(f"Testing query: {test_query}")
    print("=" * 60)

    try:
        # Run the agent
        result = await agent.arun(test_query)

        print("Agent Result:")
        print(f"- Answer: {result.get('answer', 'No answer')[:100]}...")
        print(
            f"- Iterations completed: {result.get('iterations_completed', 'Unknown')}"
        )
        print(f"- Query type: {result.get('query_type', 'Unknown')}")
        print(f"- Skipped iterations: {result.get('skipped_iterations', False)}")

        if result.get("sufficiency_assessment"):
            assessment = result["sufficiency_assessment"]
            print(f"- Sufficiency assessment:")
            print(f"  - Is sufficient: {assessment.get('is_sufficient', False)}")
            print(f"  - Final decision: {assessment.get('final_decision', False)}")
            print(f"  - Reasoning: {assessment.get('reasoning', 'No reasoning')}")
            print(f"  - Confidence: {assessment.get('confidence', 0.0)}")

        # Check if the fix worked
        if result.get("iterations_completed") == 0 and result.get("skipped_iterations"):
            print(
                "\n✅ SUCCESS: Agent correctly skipped iterations for sufficient results!"
            )
        elif result.get("iterations_completed") == 1:
            print(
                "\n⚠️  PARTIAL: Agent completed 1 iteration (acceptable for sufficient results)"
            )
        else:
            print(
                f"\n❌ ISSUE: Agent completed {result.get('iterations_completed')} iterations despite sufficient results"
            )

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


async def test_insufficient_results():
    """Test that the agent correctly handles insufficient initial results."""

    # Mock OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "test-key"))

    # Create a mock retriever that returns poor results
    class MockRetriever:
        async def search(
            self,
            user_query,
            num_subqueries=2,
            initial_candidate_pool_size=20,
            top_k_kg_entities=10,
        ):
            return {
                "llm_formatted_context": f"""
                Original Query: {user_query}

                --- Results for Sub-query: "{user_query}" ---

                Vector Search Results (Chunks):
                Source ID [c_xyz789_p1_i1]: (Score: 0.45)
                Brief mention of topic.
                File: unrelated_doc.pdf, Page: 1, Chunk Index in Page: 1
                """,
                "sub_queries_results": [],
            }

    # Create agent with mock retriever
    agent = StaticResearchAgent(
        llm_client=openai_client,
        retriever=MockRetriever(),
        params={"model": "gpt-4o-mini"},
    )

    # Test query
    test_query = "What are the detailed specifications and technical requirements?"

    print(f"\nTesting insufficient results with query: {test_query}")
    print("=" * 60)

    try:
        # Run the agent
        result = await agent.arun(test_query)

        print("Agent Result:")
        print(f"- Answer: {result.get('answer', 'No answer')[:100]}...")
        print(
            f"- Iterations completed: {result.get('iterations_completed', 'Unknown')}"
        )
        print(f"- Query type: {result.get('query_type', 'Unknown')}")
        print(f"- Skipped iterations: {result.get('skipped_iterations', False)}")

        if result.get("sufficiency_assessment"):
            assessment = result["sufficiency_assessment"]
            print(f"- Sufficiency assessment:")
            print(f"  - Is sufficient: {assessment.get('is_sufficient', False)}")
            print(f"  - Final decision: {assessment.get('final_decision', False)}")
            print(f"  - Reasoning: {assessment.get('reasoning', 'No reasoning')}")
            print(f"  - Confidence: {assessment.get('confidence', 0.0)}")

        # Check if the agent proceeded with iterations for insufficient results
        if result.get("iterations_completed", 0) > 0 and not result.get(
            "skipped_iterations"
        ):
            print(
                "\n✅ SUCCESS: Agent correctly proceeded with iterations for insufficient results!"
            )
        else:
            print(
                f"\n❌ ISSUE: Agent should have proceeded with iterations for insufficient results"
            )

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests."""
    print("Testing StaticResearchAgent sufficiency logic fixes...")
    print("=" * 80)

    await test_agent_sufficiency_logic()
    await test_insufficient_results()

    print("\n" + "=" * 80)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
