#!/usr/bin/env python3
"""
Test script for the new chunk and knowledge graph viewing endpoints.
"""

import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("ROWBLAZE_API_URL", "http://localhost:8000/api")
TEST_INDEX = "test-index"


async def test_endpoints():
    """Test the new chunk and KG endpoints."""

    # Test data
    headers = {
        "Authorization": "Bearer test-token",  # Replace with actual token
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Test chunks endpoint
        print("Testing chunks endpoint...")
        try:
            response = await client.get(
                f"{API_URL}/chunks/{TEST_INDEX}", headers=headers
            )
            print(f"Chunks endpoint status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Chunks response: {data.get('total', 0)} chunks found")
            else:
                print(f"Chunks endpoint error: {response.text}")
        except Exception as e:
            print(f"Chunks endpoint exception: {e}")

        # Test knowledge graph endpoint
        print("\nTesting knowledge graph endpoint...")
        try:
            response = await client.get(
                f"{API_URL}/knowledge-graph/{TEST_INDEX}", headers=headers
            )
            print(f"KG endpoint status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                kg = data.get("knowledge_graph", {})
                print(
                    f"KG response: {data.get('total_entities', 0)} entities, {data.get('total_relationships', 0)} relationships"
                )
            else:
                print(f"KG endpoint error: {response.text}")
        except Exception as e:
            print(f"KG endpoint exception: {e}")

        # Test with file filter
        print("\nTesting with file filter...")
        try:
            response = await client.get(
                f"{API_URL}/chunks/{TEST_INDEX}",
                params={"file_name": "test.pdf"},
                headers=headers,
            )
            print(f"Filtered chunks status: {response.status_code}")
        except Exception as e:
            print(f"Filtered chunks exception: {e}")


if __name__ == "__main__":
    print("Testing chunk and knowledge graph endpoints...")
    asyncio.run(test_endpoints())
    print("Test completed.")
