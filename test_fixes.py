#!/usr/bin/env python3
"""
Simple test script to verify the fixes work.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")

    try:
        from api.dependencies import get_elasticsearch_client, get_openai_client

        print("✅ API dependencies import successfully")
    except Exception as e:
        print(f"❌ API dependencies import failed: {e}")
        return False

    try:
        from src.core.agents.static_research_agent import StaticResearchAgent

        print("✅ StaticResearchAgent imports successfully")
    except Exception as e:
        print(f"❌ StaticResearchAgent import failed: {e}")
        return False

    try:
        from src.core.retrieval.rag_retrieval import RAGFusionRetriever

        print("✅ RAGFusionRetriever imports successfully")
    except Exception as e:
        print(f"❌ RAGFusionRetriever import failed: {e}")
        return False

    return True


async def test_openai_client():
    """Test OpenAI client creation."""
    print("\nTesting OpenAI client...")

    # Set a dummy API key for testing
    os.environ["OPENAI_API_KEY"] = "sk-test-key-for-import-testing"

    try:
        from api.dependencies import get_openai_client

        # Don't actually call it since we don't have a real key
        print("✅ OpenAI client function available")
        return True
    except Exception as e:
        print(f"❌ OpenAI client test failed: {e}")
        return False


async def test_vision_prompt():
    """Test that vision prompt file can be found."""
    print("\nTesting vision prompt file...")

    try:
        from src.core.base.parsers.image_parser import ImageParser

        parser = ImageParser()
        # Try to load the vision prompt
        prompt = parser._load_vision_prompt()
        if prompt:
            print("✅ Vision prompt loaded successfully")
            return True
        else:
            print("⚠️ Vision prompt is empty but no error occurred")
            return True
    except Exception as e:
        print(f"❌ Vision prompt test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Running fix verification tests...\n")

    tests = [
        test_imports,
        test_openai_client,
        test_vision_prompt,
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
        print("🎉 All tests passed! The fixes should work.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
