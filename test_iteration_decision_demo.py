#!/usr/bin/env python3
"""
Demonstration of the improved StaticResearchAgent iteration decision logic.
This shows how GPT-4o-mini is used to decide whether to proceed to a second iteration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.agents.static_research_agent import StaticResearchAgent


async def demo_iteration_decision():
    """Demonstrate the iteration decision logic with mock scenarios."""

    print("🎯 StaticResearchAgent Iteration Decision Demo")
    print("=" * 70)
    print("This demo shows how GPT-4o-mini decides whether to proceed to iteration 2")
    print()

    try:
        agent = StaticResearchAgent()

        # Test the sufficiency assessment function directly with different scenarios
        scenarios = [
            {
                "name": "Scenario 1: Sufficient Context",
                "query": "What is the capital of France?",
                "context": """
France is a European country with a rich history and culture. Paris is the capital and largest city of France,
located in the north-central part of the country on the Seine River. Paris has been the capital since 987 AD
and is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city has a
population of over 2 million people within the city limits and over 12 million in the metropolitan area.
""",
                "expected_sufficient": True,
            },
            {
                "name": "Scenario 2: Insufficient Context",
                "query": "What is the capital of France?",
                "context": "France is a beautiful country in Europe with many cities and regions.",
                "expected_sufficient": False,
            },
            {
                "name": "Scenario 3: Complex Query - Sufficient Context",
                "query": "Analyze the economic impact of renewable energy adoption",
                "context": """
Renewable energy adoption has significant economic impacts across multiple sectors. Key economic benefits include:
1. Job creation: The renewable energy sector employs over 12 million people globally
2. Cost reduction: Solar and wind costs have decreased by 70-80% since 2010
3. Energy independence: Reduces reliance on fossil fuel imports
4. GDP growth: Studies show 0.5-1% GDP increase per year in countries with high renewable adoption
5. Investment flows: $300+ billion annually in renewable energy investments
6. Health cost savings: Reduced air pollution saves $50-100 billion in healthcare costs
Economic challenges include transition costs, grid infrastructure investments, and job displacement in fossil fuel industries.
""",
                "expected_sufficient": True,
            },
            {
                "name": "Scenario 4: Complex Query - Insufficient Context",
                "query": "Analyze the economic impact of renewable energy adoption",
                "context": "Renewable energy is becoming more popular. It includes solar and wind power.",
                "expected_sufficient": False,
            },
        ]

        print("🔍 Testing Sufficiency Assessment Logic:")
        print("-" * 50)

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{scenario['name']}:")
            print(f"Query: {scenario['query']}")
            print(f"Context: {scenario['context'][:100]}...")
            print(f"Expected Sufficient: {scenario['expected_sufficient']}")

            # Test the sufficiency assessment
            assessment = await agent._enhanced_sufficiency_check(
                scenario["context"], scenario["query"], "factual_lookup"
            )

            is_sufficient = assessment.get("final_decision", False)
            reasoning = assessment.get("reasoning", "No reasoning")
            confidence = assessment.get("confidence", 0.0)

            print(f"🤖 GPT-4o-mini Assessment:")
            print(f"   - Sufficient: {is_sufficient}")
            print(f"   - Confidence: {confidence:.2f}")
            print(f"   - Reasoning: {reasoning}")

            # Check if assessment matches expectation
            if is_sufficient == scenario["expected_sufficient"]:
                print(f"✅ Assessment matches expectation")
            else:
                print(f"⚠️  Assessment differs from expectation")

            # Explain what would happen in the agent
            if is_sufficient:
                print(
                    f"📋 Agent Decision: Would skip iteration 2 and generate final answer"
                )
            else:
                print(
                    f"📋 Agent Decision: Would proceed to iteration 2 for more information"
                )

        print(f"\n🎯 Key Improvements Made:")
        print(
            f"   ✅ GPT-4o-mini evaluates context sufficiency after initial retrieval"
        )
        print(f"   ✅ Agent skips iteration 2 if GPT-4o-mini deems context sufficient")
        print(f"   ✅ Agent proceeds to iteration 2 only if more information is needed")
        print(f"   ✅ Balanced assessment criteria (not too strict, not too lenient)")
        print(f"   ✅ Post-iteration 1 assessment to decide on iteration 2")

        print(f"\n📊 Decision Flow:")
        print(f"   1. Initial retrieval performed")
        print(f"   2. GPT-4o-mini evaluates context sufficiency")
        print(f"   3. If sufficient → Generate final answer (skip iteration 2)")
        print(f"   4. If insufficient → Proceed to iteration 1 with tools")
        print(f"   5. After iteration 1 → GPT-4o-mini re-evaluates")
        print(f"   6. If now sufficient → Generate final answer")
        print(f"   7. If still insufficient → Proceed to iteration 2")

        await agent.cleanup()

    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


async def demo_fallback_assessment():
    """Demonstrate the fallback heuristic assessment."""

    print(f"\n🔧 Fallback Heuristic Assessment Demo")
    print("=" * 50)
    print("This shows the heuristic logic when GPT-4o-mini is unavailable")

    try:
        agent = StaticResearchAgent()

        test_cases = [
            {
                "query": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or classifications.",
                "description": "Good overlap and sufficient length",
            },
            {
                "query": "What is machine learning?",
                "context": "Technology is advancing rapidly.",
                "description": "Poor overlap and short length",
            },
            {
                "query": "Explain quantum computing",
                "context": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This allows quantum computers to perform certain calculations exponentially faster than classical computers, particularly for problems involving cryptography, optimization, and simulation of quantum systems.",
                "description": "Comprehensive context with good overlap",
            },
        ]

        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test['description']}")
            print(f"Query: {test['query']}")
            print(f"Context: {test['context'][:100]}...")

            # Use the fallback assessment directly
            assessment = agent._fallback_critique_assessment(
                test["context"], test["query"]
            )

            print(f"🔧 Heuristic Assessment:")
            print(f"   - Sufficient: {assessment['is_sufficient']}")
            print(f"   - Confidence: {assessment['confidence']:.2f}")
            print(f"   - Reasoning: {assessment['reasoning']}")

        await agent.cleanup()

    except Exception as e:
        print(f"❌ Fallback demo error: {e}")


if __name__ == "__main__":
    print("🚀 Starting Iteration Decision Demo")

    # Check environment
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPEN_AI_KEY"):
        print(
            "⚠️  Warning: OPENAI_API_KEY not set. GPT-4o-mini assessment will use fallback heuristics."
        )
        print("   Set OPENAI_API_KEY to see full GPT-4o-mini decision logic.")

    # Run demos
    asyncio.run(demo_iteration_decision())
    asyncio.run(demo_fallback_assessment())

    print(f"\n✅ Demo completed!")
    print(f"\n📝 Summary:")
    print(f"   The agent now uses GPT-4o-mini to intelligently decide whether")
    print(f"   to proceed to a second iteration based on context sufficiency.")
    print(f"   This prevents unnecessary iterations when the first retrieval")
    print(f"   provides adequate information to answer the user's query.")
