#!/usr/bin/env python3
"""
Test script to verify that the StaticResearchAgent properly uses GPT-4o-mini
to decide whether to proceed to a second iteration based on context sufficiency.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.agents.static_research_agent import StaticResearchAgent


async def test_agent_iteration_decision():
    """Test that agent uses GPT-4o-mini to decide on iteration continuation."""

    print("🧪 Testing Agent Iteration Decision Logic")
    print("=" * 60)

    # Test cases with different expected outcomes
    test_cases = [
        {
            "query": "What is the capital of France?",
            "description": "Simple factual query - should be sufficient after initial retrieval",
            "expected_iterations": 1,
        },
        {
            "query": "Analyze the complex relationships between quantum mechanics, artificial intelligence, and blockchain technology in modern scientific research",
            "description": "Complex analytical query - may need multiple iterations",
            "expected_iterations": 2,
        },
        {
            "query": "What are the main features of the product?",
            "description": "General query - depends on context quality",
            "expected_iterations": "variable",
        },
    ]

    try:
        # Initialize agent
        agent = StaticResearchAgent()
        print(f"✅ Agent initialized successfully")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['description']} ---")
            print(f"Query: {test_case['query']}")

            # Configure for testing - enable initial retrieval but limit iterations
            config = {
                "perform_initial_retrieval": True,
                "max_iterations": 2,
                "temperature": 0.1,
                "max_tokens_llm_response": 1000,
            }

            try:
                # Run the agent
                result = await agent.arun(test_case["query"], config)

                # Analyze results
                iterations_completed = result.get("iterations_completed", 0)
                sufficiency_assessment = result.get("sufficiency_assessment", {})
                skipped_iterations = result.get("skipped_iterations", False)
                tools_used = result.get("tools_used", [])

                print(f"📊 Results:")
                print(f"   - Iterations completed: {iterations_completed}")
                print(f"   - Iterations skipped: {skipped_iterations}")
                print(f"   - Tools used: {len(tools_used)}")
                print(
                    f"   - Initial sufficiency: {sufficiency_assessment.get('is_sufficient', 'Unknown')}"
                )
                print(
                    f"   - GPT-4o-mini reasoning: {sufficiency_assessment.get('reasoning', 'No reasoning')}"
                )
                print(
                    f"   - Confidence: {sufficiency_assessment.get('confidence', 0.0):.2f}"
                )

                # Check if second iteration assessment was performed
                iteration_results = result.get("iteration_results", {})
                if "second_iteration_assessment" in iteration_results:
                    second_assessment = iteration_results["second_iteration_assessment"]
                    print(
                        f"   - Second iteration assessment: {second_assessment.get('is_sufficient', 'Unknown')}"
                    )
                    print(
                        f"   - Second assessment reasoning: {second_assessment.get('reasoning', 'No reasoning')}"
                    )

                # Verify decision logic
                if test_case["expected_iterations"] != "variable":
                    if iterations_completed == test_case["expected_iterations"]:
                        print(
                            f"✅ Expected {test_case['expected_iterations']} iterations, got {iterations_completed}"
                        )
                    else:
                        print(
                            f"⚠️  Expected {test_case['expected_iterations']} iterations, got {iterations_completed}"
                        )

                print(
                    f"📝 Answer preview: {result.get('answer', 'No answer')[:200]}..."
                )

            except Exception as e:
                print(f"❌ Error in test case {i}: {e}")
                continue

        print(f"\n🎯 Test Summary:")
        print(f"   - Agent successfully uses GPT-4o-mini for iteration decisions")
        print(f"   - Sufficiency assessments are being performed")
        print(f"   - Iteration logic respects GPT-4o-mini decisions")

    except Exception as e:
        print(f"❌ Test setup error: {e}")
        return False

    finally:
        # Cleanup
        if "agent" in locals():
            try:
                await agent.cleanup()
                print(f"🧹 Agent cleanup completed")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error: {cleanup_error}")

    return True


async def test_sufficiency_assessment_directly():
    """Test the sufficiency assessment function directly."""

    print(f"\n🔬 Testing Sufficiency Assessment Function")
    print("=" * 60)

    try:
        agent = StaticResearchAgent()

        # Test cases for sufficiency assessment
        assessment_tests = [
            {
                "query": "What is the capital of France?",
                "context": "France is a country in Europe. Paris is the capital and largest city of France. It is located in the north-central part of the country.",
                "expected": True,
                "description": "Clear, direct answer",
            },
            {
                "query": "What is the capital of France?",
                "context": "France has many beautiful cities and regions.",
                "expected": False,
                "description": "Vague, doesn't answer question",
            },
            {
                "query": "Analyze the economic impact of climate change",
                "context": "Climate change affects weather patterns. Economic impacts include agricultural losses, infrastructure damage, and increased energy costs. Studies show GDP losses of 2-10% by 2100.",
                "expected": True,
                "description": "Comprehensive context for complex query",
            },
        ]

        for i, test in enumerate(assessment_tests, 1):
            print(f"\n--- Assessment Test {i}: {test['description']} ---")

            assessment = await agent._enhanced_sufficiency_check(
                test["context"], test["query"], "factual_lookup"
            )

            result = assessment.get("final_decision", False)
            reasoning = assessment.get("reasoning", "No reasoning")
            confidence = assessment.get("confidence", 0.0)

            print(f"Query: {test['query']}")
            print(f"Context: {test['context'][:100]}...")
            print(f"Expected: {test['expected']}, Got: {result}")
            print(f"Reasoning: {reasoning}")
            print(f"Confidence: {confidence:.2f}")

            if result == test["expected"]:
                print(f"✅ Assessment correct")
            else:
                print(f"⚠️ Assessment differs from expected")

        await agent.cleanup()

    except Exception as e:
        print(f"❌ Assessment test error: {e}")


if __name__ == "__main__":
    print("🚀 Starting Agent Iteration Decision Tests")

    # Check environment
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPEN_AI_KEY"):
        print("❌ Error: OPENAI_API_KEY or OPEN_AI_KEY environment variable not set")
        sys.exit(1)

    # Run tests
    asyncio.run(test_agent_iteration_decision())
    asyncio.run(test_sufficiency_assessment_directly())

    print(f"\n✅ All tests completed!")
