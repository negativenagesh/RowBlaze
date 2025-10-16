#!/usr/bin/env python3
"""
Test script to verify that Agentic RAG mode is working correctly.
"""

import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000/api"


async def test_agentic_rag():
    """Test the agentic RAG endpoint"""

    # Test payload
    payload = {
        "question": "What are the main features of the system?",
        "index_name": "test-index",
        "top_k_chunks": 5,
        "enable_references_citations": True,
        "deep_research": False,
        "auto_chunk_sizing": True,
        "model": "gpt-4o-mini",
        "max_tokens": 16384,
        "max_iterations": 2,
        "use_agent": True,
        "query_complexity_analysis": True,
    }

    # Mock auth token (you'll need to replace this with a real token)
    headers = {"Authorization": "Bearer test-token", "Content-Type": "application/json"}

    try:
        print("🧪 Testing Agentic RAG endpoint...")
        print(f"📡 Calling: {API_URL}/agent-query")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{API_URL}/agent-query", json=payload, headers=headers
            )

            print(f"📊 Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("✅ Agentic RAG response received!")
                print(f"🤖 Answer: {result.get('answer', 'No answer')[:200]}...")

                metadata = result.get("metadata", {})
                print(f"🔧 Tools used: {metadata.get('tools_used', [])}")
                print(f"🔄 Iterations: {metadata.get('iterations_completed', 0)}")
                print(f"🎯 Query type: {metadata.get('query_type', 'unknown')}")
                print(f"📈 Dynamic top-k: {metadata.get('dynamic_top_k', 'unknown')}")

                if metadata.get("tools_used"):
                    print("🎉 SUCCESS: Tools were used in agentic mode!")
                else:
                    print(
                        "⚠️  WARNING: No tools were used - this might indicate an issue"
                    )

            else:
                print(f"❌ Error: {response.status_code}")
                print(f"📄 Response: {response.text}")

    except Exception as e:
        print(f"💥 Exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_agentic_rag())
