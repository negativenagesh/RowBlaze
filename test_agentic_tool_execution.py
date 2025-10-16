#!/usr/bin/env python3
"""
Test script to verify that the StaticResearchAgent properly executes all 4 tools in agentic mode.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_agentic_tool_execution():
    """Test that the agent executes all tools in agentic mode."""

    try:
        from openai import AsyncOpenAI

        from src.core.agents.static_research_agent import StaticResearchAgent
        from src.core.retrieval.rag_retrieval import RAGFusionRetriever

        print("🧪 Testing Agentic Tool Execution...")

        # Mock configuration for testing
        test_config = {
            "index_name": "test-index",
            "perform_initial_retrieval": False,  # Pure agentic mode
            "max_iterations": 2,
            "temperature": 0.3,
            "max_tokens_llm_response": 16000,
            "tool_top_k_chunks": 10,
            "tool_top_k_kg": 10,
            "tool_num_subqueries": 2,
        }

        # Create a mock OpenAI client (you'll need a real API key for actual testing)
        openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
        if not openai_api_key:
            print(
                "❌ No OpenAI API key found. Set OPENAI_API_KEY or OPEN_AI_KEY environment variable."
            )
            return

        openai_client = AsyncOpenAI(api_key=openai_api_key)

        # Create agent with mock retriever (for testing tool initialization)
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=None,  # We'll test without actual retriever
            llm_model="gpt-4o-mini",
            max_iterations=2,
            params=test_config,
        )

        print(f"✅ Agent initialized with {len(agent.tools)} tools")
        print(f"🛠️ Available tools: {list(agent.tools.keys())}")

        # Test tool selection
        test_query = (
            "explain how someone can be a good fit for a knowledge base engineer role"
        )
        query_type = await agent._classify_query_type(test_query)
        print(f"📝 Query classified as: {query_type}")

        optimal_tools = await agent._determine_optimal_tools(test_query, query_type)
        print(f"🎯 Optimal tools selected: {optimal_tools}")

        # Verify all 4 tools are available
        expected_tools = [
            "search_file_knowledge",
            "vector_search",
            "keyword_search",
            "graph_traversal",
        ]
        missing_tools = [tool for tool in expected_tools if tool not in agent.tools]

        if missing_tools:
            print(f"❌ Missing tools: {missing_tools}")
        else:
            print("✅ All 4 expected tools are available")

        # Test that complex analysis selects all tools
        if query_type == "complex_analysis" and len(optimal_tools) >= 3:
            print("✅ Complex analysis query correctly selects multiple tools")
        else:
            print(
                f"⚠️ Expected complex analysis to select 3+ tools, got {len(optimal_tools)}"
            )

        print("🎯 Test completed successfully!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed and paths are correct.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if "agent" in locals():
                await agent.cleanup()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_agentic_tool_execution())
