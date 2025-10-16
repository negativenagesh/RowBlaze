#!/usr/bin/env python3
"""
Comprehensive test comparing Normal RAG vs Pure Agentic RAG modes
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


async def test_normal_rag_mode():
    """Test Normal RAG mode (with initial retrieval)"""
    print("\n" + "=" * 60)
    print("🚀 TESTING NORMAL RAG MODE")
    print("=" * 60)

    try:
        from openai import AsyncOpenAI

        from src.core.agents.static_research_agent import StaticResearchAgent
        from src.core.retrieval.rag_retrieval import RAGFusionRetriever

        # Create OpenAI client
        openai_client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_KEY"))

        test_query = "What are the main features of a good software system?"

        # Normal RAG mode configuration
        agent_params = {
            "question": test_query,
            "top_k_chunks": 10,
            "model": "gpt-4o-mini",
            "index_name": "test-index",
        }

        agent_config = {
            "index_name": "test-index",
            "perform_initial_retrieval": True,  # Normal mode WITH initial retrieval
            "max_iterations": 2,
            "temperature": 0.3,
        }

        print(f"📊 Normal RAG Mode Configuration:")
        print(
            f"   - perform_initial_retrieval: {agent_config['perform_initial_retrieval']}"
        )
        print(f"   - Has retriever: Will create RAGFusionRetriever")

        # Create retriever for normal mode
        retriever = RAGFusionRetriever(
            params=agent_params,
            config=agent_config,
            es_client=None,  # No ES for testing
            aclient_openai=openai_client,
        )

        # Create agent with retriever (normal mode)
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=retriever,  # Has retriever for initial search
            llm_model="gpt-4o-mini",
            max_iterations=2,
            params=agent_params,
        )

        print(f"🛠️ Agent tools: {list(agent.tools.keys())}")

        # Run normal RAG mode
        result = await agent.arun(test_query, agent_config)

        print(f"\n📊 Normal RAG Results:")
        print(f"   Answer length: {len(result.get('answer', ''))}")
        print(f"   Iterations: {result.get('iterations_completed', 0)}")
        print(f"   Tools used: {len(result.get('tools_used', []))}")
        print(f"   Mode: Normal RAG with initial retrieval")

        await agent.cleanup()
        return result

    except Exception as e:
        print(f"💥 Normal RAG Error: {e}")
        return None


async def test_pure_agentic_mode():
    """Test Pure Agentic RAG mode (no initial retrieval)"""
    print("\n" + "=" * 60)
    print("🧠 TESTING PURE AGENTIC RAG MODE")
    print("=" * 60)

    try:
        from openai import AsyncOpenAI

        from src.core.agents.static_research_agent import StaticResearchAgent

        # Create OpenAI client
        openai_client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_KEY"))

        test_query = "What are the main features of a good software system?"

        # Pure agentic mode configuration
        agent_params = {
            "question": test_query,
            "top_k_chunks": 10,
            "model": "gpt-4o-mini",
            "index_name": "test-index",
        }

        agent_config = {
            "index_name": "test-index",
            "perform_initial_retrieval": False,  # Pure agentic mode - NO initial retrieval
            "max_iterations": 2,
            "temperature": 0.3,
            "tool_top_k_chunks": 10,
        }

        print(f"🎯 Pure Agentic Mode Configuration:")
        print(
            f"   - perform_initial_retrieval: {agent_config['perform_initial_retrieval']}"
        )
        print(f"   - No retriever: Agent uses tools only")

        # Create agent without retriever (pure agentic mode)
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=None,  # NO retriever - pure agentic
            llm_model="gpt-4o-mini",
            max_iterations=2,
            params=agent_params,
        )

        print(f"🛠️ Agent tools: {list(agent.tools.keys())}")

        # Run pure agentic mode
        result = await agent.arun(test_query, agent_config)

        print(f"\n📊 Pure Agentic Results:")
        print(f"   Answer length: {len(result.get('answer', ''))}")
        print(f"   Iterations: {result.get('iterations_completed', 0)}")
        print(f"   Tools used: {len(result.get('tools_used', []))}")
        print(f"   Mode: Pure Agentic with tool-based retrieval")

        if result.get("tools_used"):
            print(f"   🎯 Tools executed:")
            for tool in result["tools_used"]:
                print(f"      - {tool['tool_name']}: {tool['parameters']}")

        await agent.cleanup()
        return result

    except Exception as e:
        print(f"💥 Pure Agentic Error: {e}")
        return None


async def main():
    """Compare Normal RAG vs Pure Agentic RAG modes"""
    print("🔬 RAG MODES COMPARISON TEST")
    print(
        "This test demonstrates the difference between Normal RAG and Pure Agentic RAG"
    )

    # Test both modes
    normal_result = await test_normal_rag_mode()
    agentic_result = await test_pure_agentic_mode()

    # Comparison summary
    print("\n" + "=" * 60)
    print("📊 DETAILED COMPARISON SUMMARY")
    print("=" * 60)

    print("\n🚀 NORMAL RAG MODE:")
    print("  • Uses RAGFusionRetriever for initial document retrieval")
    print(
        "  • Process: Initial retrieval → sufficiency check → optional tools → answer"
    )
    print("  • May skip tool usage if initial retrieval is deemed sufficient")
    print("  • Hybrid approach: retrieval + optional agentic enhancement")
    if normal_result:
        print(
            f"  • Result: {normal_result.get('iterations_completed', 0)} iterations, {len(normal_result.get('tools_used', []))} tools used"
        )

    print("\n🧠 PURE AGENTIC RAG MODE:")
    print("  • NO initial retrieval - starts with empty context")
    print(
        "  • Process: Query analysis → tool selection → iterative tool usage → synthesis"
    )
    print("  • ALWAYS uses tools for information gathering")
    print("  • Pure agentic approach: complete tool-based research")
    if agentic_result:
        print(
            f"  • Result: {agentic_result.get('iterations_completed', 0)} iterations, {len(agentic_result.get('tools_used', []))} tools used"
        )

    print("\n🎯 KEY ARCHITECTURAL DIFFERENCES:")
    print("   Normal RAG = RAGFusionRetriever + Optional Agent Tools")
    print("   Pure Agentic = Agent Tools Only (No Initial Retrieval)")

    print("\n🔧 WHEN TO USE EACH MODE:")
    print("   Normal RAG: When you want fast retrieval with optional enhancement")
    print("   Pure Agentic: When you want complete tool-based research and analysis")

    if agentic_result and agentic_result.get("tools_used"):
        print(f"\n✅ PURE AGENTIC MODE IS WORKING CORRECTLY!")
        print(
            f"   The agent is operating in pure agentic mode with tool-based retrieval."
        )
        print(
            f"   No initial retrieval was performed - all information came from tools."
        )
    else:
        print(f"\n⚠️ Pure agentic mode may need debugging")

    print(f"\n🎉 COMPARISON TEST COMPLETE!")


if __name__ == "__main__":
    asyncio.run(main())
