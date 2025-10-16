#!/usr/bin/env python3
"""
Test script to verify retrieval fixes.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_model_parameter():
    """Test that model parameters are handled correctly."""
    print("Testing model parameter handling...")

    # Set environment variables
    os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
    os.environ["OPENAI_API_KEY"] = "test-key"

    try:
        from src.core.agents.static_research_agent import (
            DEFAULT_LLM_MODEL,
            StaticResearchAgent,
        )

        print(f"✅ DEFAULT_LLM_MODEL: {DEFAULT_LLM_MODEL}")

        # Test agent initialization with params
        params = {"model": "gpt-4o-mini", "question": "test"}
        agent = StaticResearchAgent(params=params)

        print(f"✅ Agent llm_model: {agent.llm_model}")
        print(f"✅ Agent config model: {agent.config.get('model')}")

        return True

    except Exception as e:
        print(f"❌ Model parameter test failed: {e}")
        return False


async def test_rag_retriever_method():
    """Test that RAGFusionRetriever has the correct method."""
    print("\nTesting RAGFusionRetriever method availability...")

    try:
        from src.core.retrieval.rag_retrieval import RAGFusionRetriever

        # Check if search method exists
        if hasattr(RAGFusionRetriever, "search"):
            print("✅ RAGFusionRetriever.search method exists")
        else:
            print("❌ RAGFusionRetriever.search method missing")
            return False

        # Check if process_query method exists (should not)
        if hasattr(RAGFusionRetriever, "process_query"):
            print("⚠️ RAGFusionRetriever.process_query method exists (unexpected)")
        else:
            print("✅ RAGFusionRetriever.process_query method correctly absent")

        return True

    except Exception as e:
        print(f"❌ RAGFusionRetriever test failed: {e}")
        return False


async def test_imports():
    """Test that all imports work correctly."""
    print("\nTesting imports...")

    try:
        from api.routes.retrieval import RAGFusionRetriever, StaticResearchAgent

        print("✅ API route imports work")

        from src.core.agents.static_research_agent import StaticResearchAgent

        print("✅ StaticResearchAgent imports work")

        from src.core.retrieval.rag_retrieval import RAGFusionRetriever

        print("✅ RAGFusionRetriever imports work")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Running retrieval fix verification tests...\n")

    tests = [
        test_imports,
        test_model_parameter,
        test_rag_retriever_method,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("🎉 All tests passed! The retrieval fixes should work.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
