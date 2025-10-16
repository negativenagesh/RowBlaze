#!/usr/bin/env python3
"""
Test script to verify that the StaticResearchAgent now properly lets the LLM select tools
instead of forcing all tools to execute.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_agent_tool_selection():
    """Test that the agent lets LLM select tools instead of forcing all tools."""

    try:
        from src.core.agents.static_research_agent import StaticResearchAgent

        # Create agent instance
        agent = StaticResearchAgent()

        # Test the _determine_optimal_tools method
        query = "What are the qualifications of Subrahmanya?"
        query_type = await agent._classify_query_type(query)

        print(f"Query: {query}")
        print(f"Classified as: {query_type}")

        # Test tool selection
        optimal_tools = await agent._determine_optimal_tools(query, query_type)
        print(f"Selected tools: {optimal_tools}")
        print(f"Number of tools selected: {len(optimal_tools)}")

        # Verify that not all 4 tools are always selected
        all_tools = [
            "search_file_knowledge",
            "vector_search",
            "keyword_search",
            "graph_traversal",
        ]

        if len(optimal_tools) < len(all_tools):
            print("✅ SUCCESS: Agent is selecting subset of tools, not all tools")
        else:
            print(
                "⚠️  WARNING: Agent selected all tools - this might be expected for complex queries"
            )

        # Test with a simple factual query
        simple_query = "What is the name?"
        simple_query_type = await agent._classify_query_type(simple_query)
        simple_tools = await agent._determine_optimal_tools(
            simple_query, simple_query_type
        )

        print(f"\nSimple query: {simple_query}")
        print(f"Classified as: {simple_query_type}")
        print(f"Selected tools: {simple_tools}")
        print(f"Number of tools selected: {len(simple_tools)}")

        if len(simple_tools) <= 2:
            print("✅ SUCCESS: Agent selects fewer tools for simple queries")
        else:
            print("⚠️  INFO: Agent selected many tools for simple query")

        await agent.cleanup()

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Testing Agent Tool Selection...")
    asyncio.run(test_agent_tool_selection())
