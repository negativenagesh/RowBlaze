#!/usr/bin/env python3
"""
Test Pure Agentic RAG Mode - Verify that agent uses tools for ALL information gathering
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


async def test_pure_agentic_mode():
    """Test that pure agentic mode uses tools for all information gathering"""
    print("🧠 Testing Pure Agentic RAG Mode")
    print("=" * 60)

    try:
        from openai import AsyncOpenAI

        from src.core.agents.static_research_agent import StaticResearchAgent

        # Create OpenAI client
        openai_client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_KEY"))

        # Test query
        test_query = "What are the main features of a good software system?"

        # Agent parameters for pure agentic mode
        agent_params = {
            "question": test_query,
            "top_k_chunks": 10,
            "enable_references_citations": True,
            "deep_research": False,
            "model": "gpt-4o-mini",
            "max_iterations": 2,
            "index_name": "test-index",
        }

        # CRITICAL: Pure agentic mode configuration
        agent_config = {
            "index_name": "test-index",
            "perform_initial_retrieval": False,  # NO initial retrieval - pure agentic
            "max_iterations": 2,
            "temperature": 0.3,
            "max_tokens_llm_response": 16000,
            "tool_top_k_chunks": 10,
            "tool_top_k_kg": 10,
            "tool_num_subqueries": 2,
        }

        print(f"🎯 Creating agent in PURE AGENTIC MODE")
        print(
            f"   - perform_initial_retrieval: {agent_config['perform_initial_retrieval']}"
        )
        print(f"   - Query: {test_query}")

        # Create agent without retriever (pure agentic mode)
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=None,  # No retriever for pure agentic mode
            llm_model="gpt-4o-mini",
            max_iterations=2,
            params=agent_params,
        )

        print(f"🛠️ Agent tools available: {list(agent.tools.keys())}")

        # Run agent in pure agentic mode
        print(f"\n🚀 Running agent in pure agentic mode...")
        result = await agent.arun(test_query, agent_config)

        # Analyze results
        print(f"\n📊 Pure Agentic Mode Results:")
        print(f"   Answer length: {len(result.get('answer', ''))}")
        print(f"   Query type: {result.get('query_type', 'unknown')}")
        print(f"   Iterations completed: {result.get('iterations_completed', 0)}")
        print(f"   Tools used: {len(result.get('tools_used', []))}")

        # Check if tools were used (critical for agentic mode)
        tools_used = result.get("tools_used", [])
        if tools_used:
            print(f"\n✅ SUCCESS: Pure agentic mode used {len(tools_used)} tools!")
            for i, tool in enumerate(tools_used, 1):
                print(f"   {i}. {tool['tool_name']}: {tool['parameters']}")
        else:
            print(f"\n❌ FAILURE: No tools were used in pure agentic mode!")

        # Check sufficiency assessment
        sufficiency = result.get("sufficiency_assessment", {})
        if sufficiency:
            print(f"\n🔍 Sufficiency Assessment:")
            print(f"   Sufficient: {sufficiency.get('is_sufficient', False)}")
            print(f"   Reasoning: {sufficiency.get('reasoning', 'N/A')}")
            print(f"   Confidence: {sufficiency.get('confidence', 0.0):.2f}")

        # Check iteration results
        iteration_results = result.get("iteration_results", {})
        if iteration_results:
            print(f"\n📋 Iteration Results:")
            for iteration, results in iteration_results.items():
                if isinstance(results, str):
                    print(f"   {iteration}: {len(results)} chars")
                else:
                    print(f"   {iteration}: {type(results)}")

        # Show answer preview
        answer = result.get("answer", "")
        if answer:
            print(f"\n💬 Answer Preview:")
            print(f"   {answer[:200]}...")

        # Cleanup
        await agent.cleanup()

        # Final assessment
        if tools_used and not sufficiency.get("is_sufficient", True):
            print(f"\n🎉 PURE AGENTIC MODE TEST PASSED!")
            print(f"   ✅ No initial retrieval performed")
            print(f"   ✅ Agent used {len(tools_used)} tools for information gathering")
            print(f"   ✅ Proper agentic behavior demonstrated")
        else:
            print(f"\n⚠️ PURE AGENTIC MODE TEST NEEDS REVIEW")
            print(f"   Tools used: {len(tools_used)}")
            print(f"   Initial sufficient: {sufficiency.get('is_sufficient', False)}")

    except Exception as e:
        print(f"💥 Error in pure agentic mode test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_pure_agentic_mode())
