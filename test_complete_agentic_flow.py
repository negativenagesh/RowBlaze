#!/usr/bin/env python3
"""
Complete test of the agentic RAG flow to verify all 4 tools are executed.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_complete_agentic_flow():
    """Test the complete agentic flow with tool execution."""

    try:
        from openai import AsyncOpenAI

        from src.core.agents.static_research_agent import StaticResearchAgent

        print("🧪 Testing Complete Agentic Flow...")

        # Check for API key
        openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
        if not openai_api_key:
            print(
                "❌ No OpenAI API key found. Set OPENAI_API_KEY or OPEN_AI_KEY environment variable."
            )
            return

        openai_client = AsyncOpenAI(api_key=openai_api_key)

        # Test configuration for pure agentic mode
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

        # Create agent
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=None,  # Testing without actual retriever
            llm_model="gpt-4o-mini",
            max_iterations=2,
            params=test_config,
        )

        print(f"✅ Agent initialized successfully")

        # Test query that should trigger all tools
        test_query = "explain in depth how subrahmanya can be a good fit for knowledge base engineer role?"

        print(f"📝 Testing query: {test_query}")

        # Test the complete agentic flow
        try:
            result = await agent.arun(test_query, test_config)

            print(f"\n🎯 Agentic Flow Results:")
            print(f"Query Type: {result.get('query_type', 'Unknown')}")
            print(f"Iterations Completed: {result.get('iterations_completed', 0)}")
            print(f"Tools Used: {len(result.get('tools_used', []))}")

            tools_used = result.get("tools_used", [])
            if tools_used:
                print(f"🛠️ Tools executed:")
                for i, tool in enumerate(tools_used, 1):
                    print(
                        f"  {i}. {tool.get('tool_name', 'Unknown')} (Iteration {tool.get('iteration', 'N/A')})"
                    )
            else:
                print("⚠️ No tools were executed")

            # Check if all expected tools were used
            expected_tools = [
                "search_file_knowledge",
                "vector_search",
                "keyword_search",
                "graph_traversal",
            ]
            executed_tool_names = [tool.get("tool_name") for tool in tools_used]

            missing_tools = [
                tool for tool in expected_tools if tool not in executed_tool_names
            ]
            if missing_tools:
                print(f"⚠️ Missing tool executions: {missing_tools}")
            else:
                print(f"✅ All {len(expected_tools)} expected tools were executed!")

            # Check iteration results
            iteration_results = result.get("iteration_results", {})
            if iteration_results:
                print(
                    f"📊 Iteration Results: {len(iteration_results)} iterations with results"
                )
                for iteration, results in iteration_results.items():
                    if isinstance(results, str) and len(results) > 0:
                        print(
                            f"  Iteration {iteration}: {len(results)} characters of results"
                        )

            # Show final answer preview
            answer = result.get("answer", "")
            if answer:
                print(f"\n📋 Final Answer Preview (first 200 chars):")
                print(f"{answer[:200]}...")
            else:
                print("⚠️ No final answer generated")

        except Exception as e:
            print(f"❌ Error during agentic flow: {e}")
            import traceback

            traceback.print_exc()

        print("\n🎯 Test completed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
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
    asyncio.run(test_complete_agentic_flow())
