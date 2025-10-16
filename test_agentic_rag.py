#!/usr/bin/env python3
"""
Test script to debug the Agentic RAG system
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


async def test_agentic_rag():
    """Test the agentic RAG system directly"""
    try:
        print("🔧 Testing Agentic RAG system...")

        # Import required modules
        from openai import AsyncOpenAI

        from src.core.agents.static_research_agent import StaticResearchAgent
        from src.core.retrieval.rag_retrieval import RAGFusionRetriever

        print("✅ Imports successful")

        # Create OpenAI client
        openai_client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_KEY"))
        print("✅ OpenAI client created")

        # Create agent parameters
        agent_params = {
            "question": "What are the main features of a good software system?",
            "top_k_chunks": 10,
            "enable_references_citations": True,
            "deep_research": False,
            "auto_chunk_sizing": True,
            "model": "gpt-4o-mini",
            "max_tokens": 16384,
            "max_iterations": 2,
            "index_name": "test-index",
        }

        agent_config = {
            "index_name": "test-index",
            "perform_initial_retrieval": False,  # Skip initial retrieval for testing
            "max_iterations": 1,
            "temperature": 0.3,
            "max_tokens_llm_response": 1000,
        }

        print("🤖 Creating StaticResearchAgent...")

        # Create agent
        agent = StaticResearchAgent(
            llm_client=openai_client,
            retriever=None,  # No retriever for this test
            llm_model="gpt-4o-mini",
            max_iterations=1,
            params=agent_params,
        )

        print(f"✅ Agent created with {len(agent.tools)} tools")
        print(f"🛠️ Available tools: {list(agent.tools.keys())}")

        # Test tool initialization
        for tool_name, tool in agent.tools.items():
            print(f"🔧 Tool {tool_name}: {type(tool).__name__}")
            if hasattr(tool, "context"):
                print(f"   Context set: {tool.context is not None}")

        # Test agent execution
        print("🚀 Running agent...")
        result = await agent.arun(
            "What are the main features of a good software system?", agent_config
        )

        print("📊 Agent Result:")
        print(f"  Answer length: {len(result.get('answer', ''))}")
        print(f"  Tools used: {len(result.get('tools_used', []))}")
        print(f"  Iterations: {result.get('iterations_completed', 0)}")
        print(f"  Query type: {result.get('query_type', 'unknown')}")

        if result.get("tools_used"):
            print("🎉 SUCCESS: Agent used tools!")
            for tool in result["tools_used"]:
                print(f"  - {tool['tool_name']}: {tool['parameters']}")
        else:
            print("⚠️ Agent did not use tools")

        # Cleanup
        await agent.cleanup()

    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_agentic_rag())
