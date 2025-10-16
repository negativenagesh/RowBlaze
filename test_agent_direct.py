#!/usr/bin/env python3
"""
Direct test of the StaticResearchAgent without going through the API.
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


async def test_agent_directly():
    """Test the agent directly without API calls"""

    try:
        print("🔧 Testing agent import...")
        # Import the agent
        from src.core.agents.static_research_agent import StaticResearchAgent

        print("✅ Agent imported successfully")

        print("🤖 Creating StaticResearchAgent without OpenAI client...")
        # Test without OpenAI client first to see initialization
        agent = StaticResearchAgent(
            llm_client=None,  # This should create its own client
            retriever=None,
            llm_model="gpt-4o-mini",
            max_iterations=2,
            params={"model": "gpt-4o-mini"},
        )

        print("✅ Agent created successfully")
        print(f"🛠️ Agent has {len(agent.tools)} tools")

        # Don't run the actual query to avoid API calls
        print("🎉 SUCCESS: Agent initialization works!")

    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_agent_directly())
