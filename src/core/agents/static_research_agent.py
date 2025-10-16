import asyncio
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from src.core.base.abstractions import (
        AggregateSearchResult,
        ChunkSearchResult,
        KGEntity,
        KGRelationship,
        KGSearchResult,
    )
    from src.core.retrieval.rag_retrieval import RAGFusionRetriever
    from src.core.tools.KeywordSearch import KeywordSearchTool
    from src.core.tools.search_file_knowledge import SearchFileKnowledgeTool
    from src.core.tools.vectorsearch import VectorSearchTool
except ImportError as e:
    print(
        f"ImportError in static_research_agent.py: {e}. Please ensure all dependencies are correctly placed and __init__.py files exist."
    )
    # Try alternative import paths for when called from different contexts
    try:
        from ...core.base.abstractions import (
            AggregateSearchResult,
            ChunkSearchResult,
            KGEntity,
            KGRelationship,
            KGSearchResult,
        )
        from ..retrieval.rag_retrieval import RAGFusionRetriever
        from ..tools.KeywordSearch import KeywordSearchTool
        from ..tools.search_file_knowledge import SearchFileKnowledgeTool
        from ..tools.vectorsearch import VectorSearchTool
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        raise

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_ITERATIONS = 2  # Changed from 5 to 2
DEFAULT_AGENT_CONFIG_PATH = (
    Path(__file__).parent.parent / "prompts" / "static_research_agent.yaml"
)

# NEW: Add API configuration
API_BASE_URL = os.getenv("ROWBLAZE_API_URL", "http://localhost:8000/api")


# Add a simple cache for similar queries
class SimpleCache:
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value


# Abstract tool interface with registration system
class AbstractTool:
    """Base abstract class for all agent tools."""

    name = "abstract_tool"
    description = "Base tool class"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to process",
            }
        },
        "required": ["query"],
    }

    def __init__(self):
        self.context = None

    def set_context(self, context):
        """Set the agent context for this tool."""
        self.context = context

    async def _call(self, **kwargs) -> str:
        """Abstract method to be implemented by concrete tools."""
        raise NotImplementedError("Tools must implement _call method")

    @property
    def supported_query_types(self) -> List[str]:
        """Returns list of query types this tool is optimized for."""
        return [
            "factual_lookup",
            "summary_extraction",
            "comparison",
            "complex_analysis",
        ]

    @property
    def priority(self) -> int:
        """Return priority of this tool (lower is higher priority)."""
        return 100


# Enhanced tool registry with metadata
class ToolRegistry:
    """Registry for tools with metadata about capabilities."""

    def __init__(self):
        self.tools = {}
        self.tool_metadata = {}

    def register(
        self, tool_instance: AbstractTool, metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a tool with optional metadata."""
        self.tools[tool_instance.name] = tool_instance
        self.tool_metadata[tool_instance.name] = metadata or {
            "supported_query_types": tool_instance.supported_query_types,
            "priority": tool_instance.priority,
            "capabilities": [],
        }

    def get_tools_for_query_type(self, query_type: str) -> List[str]:
        """Get tool names optimized for a given query type."""
        return [
            name
            for name, metadata in self.tool_metadata.items()
            if query_type in metadata.get("supported_query_types", [])
        ]

    def get_tool(self, name: str) -> Optional[AbstractTool]:
        """Get tool by name."""
        return self.tools.get(name)


class StaticResearchAgent:
    def __init__(
        self,
        llm_client: Optional[AsyncOpenAI] = None,
        retriever: Optional[RAGFusionRetriever] = None,  # Add this
        config_path: Optional[Union[str, Path]] = None,
        llm_model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,  # Add params as optional
    ):
        if llm_client is None:
            openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found and llm_client not provided."
                )
            self.llm_client = AsyncOpenAI(api_key=openai_api_key)
        else:
            self.llm_client = llm_client

        # Track if we own the llm_client for cleanup
        self._owns_llm_client = llm_client is None

        self.config_path = (
            Path(config_path) if config_path else DEFAULT_AGENT_CONFIG_PATH
        )
        self.config = self._load_config()
        # Merge params into config if provided
        if params:
            self.config.update(params)

        # Use params first, then llm_model parameter, then config, then default
        model_from_params = params.get("model") if params else None
        self.llm_model = (
            model_from_params
            or llm_model
            or self.config.get("llm_model", DEFAULT_LLM_MODEL)
        )
        self.max_iterations = max_iterations or self.config.get(
            "max_iterations", DEFAULT_MAX_ITERATIONS
        )

        self.system_prompt_template = self._load_prompt_template_from_config(
            "static_research_agent"
        )
        print(
            f"📝 System prompt template loaded: {len(self.system_prompt_template)} chars"
        )
        if not self.system_prompt_template:
            print("⚠️ WARNING: System prompt template is empty!")

        # Initialize only the 3 imported tools with proper error handling
        self.tools = {}

        try:
            self.tools["search_file_knowledge"] = SearchFileKnowledgeTool()
            print(f"✅ Initialized SearchFileKnowledgeTool")
        except Exception as e:
            print(f"❌ Warning: Could not initialize SearchFileKnowledgeTool: {e}")

        try:
            self.tools["vector_search"] = VectorSearchTool()
            print(f"✅ Initialized VectorSearchTool")
        except Exception as e:
            print(f"❌ Warning: Could not initialize VectorSearchTool: {e}")

        try:
            self.tools["keyword_search"] = KeywordSearchTool()
            print(f"✅ Initialized KeywordSearchTool")
        except Exception as e:
            print(f"❌ Warning: Could not initialize KeywordSearchTool: {e}")

        # NEW: Initialize graph traversal tool
        try:
            self.tools["graph_traversal"] = GraphTraversalTool()
            print(f"✅ Initialized GraphTraversalTool")
        except Exception as e:
            print(f"❌ Warning: Could not initialize GraphTraversalTool: {e}")

        print(f"🔧 Total tools initialized: {len(self.tools)}")
        print(f"🛠️ Available tools: {list(self.tools.keys())}")

        # Verify all expected tools are present
        expected_tools = [
            "search_file_knowledge",
            "vector_search",
            "keyword_search",
            "graph_traversal",
        ]
        missing_tools = [tool for tool in expected_tools if tool not in self.tools]
        if missing_tools:
            print(f"⚠️ Missing expected tools: {missing_tools}")
        else:
            print(f"✅ All {len(expected_tools)} expected tools are available")

        # Set context for tools if they need it
        for tool_name, tool_instance in self.tools.items():
            if hasattr(tool_instance, "set_context"):
                tool_instance.set_context(self)
                print(f"🔗 Set context for {tool_name}")
                # Verify the tool has execute method
                if hasattr(tool_instance, "execute"):
                    print(f"✅ Tool {tool_name} has execute method")
                else:
                    print(f"❌ Tool {tool_name} missing execute method")
                # Verify the context has retriever
                if hasattr(self, "retriever") and self.retriever:
                    print(f"✅ Agent has retriever for {tool_name}")
                else:
                    print(f"⚠️ Agent missing retriever for {tool_name}")
            else:
                print(f"⚠️ Tool {tool_name} does not have set_context method")

        # NEW: Store iteration results for combining
        self.iteration_results = {}

        # Add cache to agent initialization
        self.response_cache = SimpleCache()

        # Initialize tool registry
        self.tool_registry = ToolRegistry()

        # Register tools with metadata
        try:
            vector_search = VectorSearchTool()
            self.tool_registry.register(
                vector_search,
                {
                    "supported_query_types": [
                        "factual_lookup",
                        "summary_extraction",
                        "comparison",
                        "complex_analysis",
                    ],
                    "priority": 10,
                    "capabilities": ["semantic_search"],
                },
            )
        except Exception as e:
            print(f"Warning: Could not initialize VectorSearchTool: {e}")

        # Register other tools similarly...
        self.retriever = retriever  # Store retriever

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources to prevent unclosed client sessions."""
        try:
            if (
                hasattr(self, "_owns_llm_client")
                and self._owns_llm_client
                and hasattr(self, "llm_client")
            ):
                if hasattr(self.llm_client, "close"):
                    await self.llm_client.close()
                elif hasattr(self.llm_client, "aclose"):
                    await self.llm_client.aclose()
        except Exception as e:
            print(f"Error during agent cleanup: {e}")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                return {}
            return config_data
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _load_prompt_template_from_config(self, prompt_key: str) -> str:
        prompt_details = self.config.get(prompt_key, {})
        if isinstance(prompt_details, dict) and "template" in prompt_details:
            return prompt_details["template"]
        else:
            return ""

    # --- LLM + heuristic helpers (must be class methods, used by arun) ---
    async def _call_llm_api(
        self, prompt: str, model: str = None, max_tokens: int = 512
    ) -> str:
        """Simple wrapper to call the configured LLM client. Returns text or empty string on error."""
        if not getattr(self, "llm_client", None):
            return ""
        model_to_use = model or self.llm_model or DEFAULT_LLM_MODEL or "gpt-4o-mini"
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = await self.llm_client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return getattr(resp.choices[0].message, "content", "").strip() or ""
        except Exception as e:
            logging.warning(f"_call_llm_api error: {e}")
            return ""

    async def _llm_critique_initial_results(
        self, initial_context: str, query: str
    ) -> Dict[str, Any]:
        """Uses GPT-4o-mini to evaluate whether the initial context is sufficient to answer the query."""
        if not initial_context or initial_context.strip().lower().startswith(
            "no initial"
        ):
            return {
                "is_sufficient": False,
                "reasoning": "No initial context available.",
                "confidence": 0.0,
                "missing_information": ["No retrieved results"],
                "final_decision": False,
            }

        critique_prompt = f"""
You are an expert evaluator analyzing whether retrieved information is sufficient to answer a user's query.

Your task is to determine if the provided context contains enough information to give a comprehensive answer to the user's question. Be balanced in your assessment - not too strict, not too lenient.

User Query: "{query}"

Retrieved Context:
{initial_context}

Evaluation Criteria:
1. Does the context directly address the main question asked?
2. Is there enough relevant information to provide a meaningful answer?
3. Are the key aspects of the query covered in the context?
4. Would the user be satisfied with an answer based on this context?
5. Consider: Simple factual queries need less context than complex analytical questions

Guidelines:
- Mark as SUFFICIENT if the context adequately addresses the query, even if additional details could be helpful
- Mark as INSUFFICIENT only if the context lacks key information needed to answer the query
- Consider the query complexity: simple questions need less context than complex analysis
- Focus on whether a reasonable answer can be constructed from the available information

Provide your assessment as a JSON object with these exact keys:
{{
    "is_sufficient": boolean,
    "reasoning": "brief explanation of your assessment",
    "confidence": float between 0.0 and 1.0,
    "missing_information": ["list", "of", "missing", "aspects"],
    "final_decision": boolean (same as is_sufficient)
}}

JSON Assessment:"""

        try:
            response_text = await self._call_llm_api(
                critique_prompt, model="gpt-4o-mini", max_tokens=300
            )

            if not response_text:
                return self._fallback_critique_assessment(initial_context, query)

            # Extract and parse JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)

                # Validate and normalize the response
                result = {
                    "is_sufficient": bool(parsed.get("is_sufficient", False)),
                    "reasoning": str(parsed.get("reasoning", "No reasoning provided")),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "missing_information": parsed.get("missing_information", []),
                    "final_decision": bool(
                        parsed.get("final_decision", parsed.get("is_sufficient", False))
                    ),
                }

                print(
                    f"LLM critique: {'Sufficient' if result['is_sufficient'] else 'Insufficient'} (confidence: {result['confidence']:.2f})"
                )
                return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Failed to parse LLM critique response: {e}")
        except Exception as e:
            print(f"Error in LLM critique: {e}")

        return self._fallback_critique_assessment(initial_context, query)

    def _fallback_critique_assessment(
        self, initial_context: str, query: str
    ) -> Dict[str, Any]:
        """Heuristic fallback for sufficiency assessment - balanced approach."""
        if not initial_context or len(initial_context.strip()) < 100:
            return {
                "is_sufficient": False,
                "reasoning": "Context is missing or too short.",
                "confidence": 0.2,
                "missing_information": ["More detailed content required"],
                "final_decision": False,
            }

        # Analyze query-context overlap
        query_words = set([w.lower() for w in re.findall(r"\w+", query)])
        context_words = set([w.lower() for w in re.findall(r"\w+", initial_context)])
        overlap = len(query_words.intersection(context_words))
        overlap_ratio = overlap / len(query_words) if query_words else 0

        # Balanced criteria for sufficiency assessment
        min_overlap_ratio = 0.5  # 50% overlap required
        min_context_length = 300  # Minimum 300 chars for basic sufficiency
        good_context_length = 800  # 800+ chars indicates comprehensive context

        # Check basic requirements
        if overlap_ratio < min_overlap_ratio:
            return {
                "is_sufficient": False,
                "reasoning": f"Low lexical overlap between query and context ({overlap_ratio:.2%} < {min_overlap_ratio:.0%}). Additional search needed.",
                "confidence": 0.3,
                "missing_information": ["More relevant context via targeted search"],
                "final_decision": False,
            }

        if len(initial_context.strip()) < min_context_length:
            return {
                "is_sufficient": False,
                "reasoning": f"Context too short for adequate answer ({len(initial_context.strip())} < {min_context_length} chars). More information needed.",
                "confidence": 0.4,
                "missing_information": ["More detailed content"],
                "final_decision": False,
            }

        # Assess sufficiency based on context quality
        context_length = len(initial_context.strip())

        # High quality context - likely sufficient
        if context_length >= good_context_length and overlap_ratio >= 0.7:
            confidence = min(0.9, 0.6 + overlap_ratio * 0.3)
            return {
                "is_sufficient": True,
                "reasoning": f"Comprehensive context with good overlap ({overlap_ratio:.2%}). Initial retrieval appears sufficient for answering the query.",
                "confidence": confidence,
                "missing_information": [],
                "final_decision": True,
            }

        # Medium quality context - check for specific indicators
        if context_length >= min_context_length and overlap_ratio >= 0.6:
            # Look for specific answer indicators in the context
            query_lower = query.lower()
            context_lower = initial_context.lower()

            # Check if context seems to directly address the query
            has_direct_answer = any(
                word in context_lower
                for word in ["answer", "result", "conclusion", "summary"]
            )
            has_specific_info = overlap_ratio > 0.65

            if has_direct_answer or has_specific_info:
                confidence = 0.6 + overlap_ratio * 0.2
                return {
                    "is_sufficient": True,
                    "reasoning": f"Context contains relevant information with {overlap_ratio:.2%} overlap. Appears to address the query adequately.",
                    "confidence": confidence,
                    "missing_information": [],
                    "final_decision": True,
                }

        # Default to insufficient if not clearly sufficient
        confidence = 0.4 + overlap_ratio * 0.2
        return {
            "is_sufficient": False,
            "reasoning": f"Context has moderate relevance ({overlap_ratio:.2%} overlap) but may benefit from additional tool-based research for completeness.",
            "confidence": confidence,
            "missing_information": ["Additional context via specialized tools"],
            "final_decision": False,
        }

    async def _enhanced_sufficiency_check(
        self, initial_context: str, query: str, query_type: str
    ) -> Dict[str, Any]:
        """Enhanced sufficiency check using LLM critique."""
        try:
            return await self._llm_critique_initial_results(initial_context, query)
        except Exception as e:
            print(f"LLM critique failed: {e}")
            return self._fallback_critique_assessment(initial_context, query)

    async def _classify_query_type(self, query: str) -> str:
        """Uses GPT-4o-mini to intelligently classify the query type for optimal processing."""
        classification_prompt = f"""
You are an expert query classifier. Analyze the user query and classify it into exactly one category.

Categories:
1. factual_lookup: Simple, direct questions asking for specific facts, numbers, dates, names, or definitions
   Examples: "What is the population of Tokyo?", "When was the company founded?", "Who is the CEO?"

2. summary_extraction: Requests for summaries, overviews, or general information about topics
   Examples: "Summarize the quarterly report", "Give me an overview of the project", "What are the main points?"

3. comparison: Questions comparing two or more items, analyzing differences or similarities
   Examples: "Compare product A vs product B", "What's the difference between X and Y?", "Which is better?"

4. complex_analysis: Multi-faceted questions requiring analysis, evaluation, or synthesis of information
   Examples: "Analyze the market trends", "What are the implications of this policy?", "Evaluate the risks"

Query to classify: "{query}"

Instructions:
- Consider the intent and complexity of the question
- Look for comparison words (vs, versus, compare, difference)
- Look for analysis words (analyze, evaluate, assess, implications)
- Look for summary words (summarize, overview, main points)
- Default to factual_lookup for simple, direct questions

Return ONLY the category name (no explanation):"""

        try:
            response = await self._call_llm_api(
                classification_prompt, model="gpt-4o-mini", max_tokens=20
            )

            # Clean and validate response
            category = response.strip().lower()
            valid_categories = [
                "factual_lookup",
                "summary_extraction",
                "comparison",
                "complex_analysis",
            ]

            if category in valid_categories:
                print(f"LLM classified query as: {category}")
                return category

            # Try to find category in response if exact match fails
            for valid_cat in valid_categories:
                if valid_cat in category:
                    print(
                        f"LLM classified query as: {valid_cat} (extracted from: {category})"
                    )
                    return valid_cat

            print(
                f"LLM returned invalid category '{category}', defaulting to factual_lookup"
            )
            return "factual_lookup"

        except Exception as e:
            print(f"Query classification failed: {e}, defaulting to factual_lookup")
            return "factual_lookup"

    async def _get_dynamic_top_k(self, query: str, query_type: str) -> int:
        """Uses GPT-4o-mini to determine optimal number of chunks based on query complexity and requirements."""

        top_k_prompt = f"""
You are an expert at determining information retrieval requirements. Analyze this query and determine how many document chunks would be needed to provide a comprehensive answer.

Query Type: {query_type}
Query: "{query}"

Guidelines:
- Simple factual questions: 3-8 chunks
- Summary requests: 8-15 chunks
- Comparison queries: 10-20 chunks
- Complex analysis: 15-25 chunks

Consider:
- Query complexity and scope
- How much context would be needed
- Whether multiple perspectives are required
- If comprehensive coverage is needed

Return ONLY a number between 3 and 25:"""

        try:
            response = await self._call_llm_api(
                top_k_prompt, model="gpt-4o-mini", max_tokens=10
            )

            # Extract number from response
            import re

            numbers = re.findall(r"\d+", response)
            if numbers:
                top_k = int(numbers[0])
                # Ensure it's within reasonable bounds
                top_k = max(3, min(25, top_k))
                print(
                    f"LLM determined optimal top_k: {top_k} for query type '{query_type}'"
                )
                return top_k

        except Exception as e:
            print(f"Error in LLM top_k determination: {e}")

        # Fallback to type-based mapping
        type_mapping = {
            "factual_lookup": 5,
            "summary_extraction": 10,
            "comparison": 15,
            "complex_analysis": 20,
        }
        fallback_k = type_mapping.get(query_type, 10)
        print(f"Using fallback top_k: {fallback_k} for query type '{query_type}'")
        return fallback_k

    def _parse_llm_tool_calls(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        tool_calls = []

        # Look for function_calls blocks
        import re

        pattern = r"<function_calls>(.*?)</function_calls>"
        matches = re.findall(pattern, response_content, re.DOTALL)

        for match in matches:
            # Look for invoke blocks within function_calls
            invoke_pattern = r"<invoke>(.*?)</invoke>"
            invoke_matches = re.findall(invoke_pattern, match, re.DOTALL)

            for invoke_match in invoke_matches:
                # Extract tool name
                tool_name_pattern = r"<tool_name>(.*?)</tool_name>"
                tool_name_match = re.search(tool_name_pattern, invoke_match)

                # Extract parameters
                params_pattern = r"<parameters>(.*?)</parameters>"
                params_match = re.search(params_pattern, invoke_match, re.DOTALL)

                if tool_name_match:
                    tool_name = tool_name_match.group(1).strip()
                    parameters = {}

                    if params_match:
                        try:
                            import json

                            parameters = json.loads(params_match.group(1).strip())
                        except:
                            # Fallback to simple parameter extraction
                            parameters = {"query": invoke_match}

                    tool_calls.append(
                        {"tool_name": tool_name, "parameters": parameters}
                    )

        return tool_calls

    async def _execute_tool_calls_in_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        config: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """Execute tool calls in parallel."""
        results = []

        print(f"🔧 Executing {len(tool_calls)} tool calls...")

        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call["tool_name"]
            parameters = tool_call["parameters"]

            print(
                f"🛠️ Executing tool {i+1}/{len(tool_calls)}: {tool_name} with params: {parameters}"
            )

            try:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    # Use the correct method name 'execute' instead of '_call'
                    if hasattr(tool, "execute"):
                        print(f"✅ Found tool {tool_name}, executing...")
                        result = await tool.execute(**parameters)
                        # Convert result to string for consistent handling
                        if hasattr(result, "llm_formatted_context"):
                            # For AggregateSearchResult objects
                            result = result.llm_formatted_context or str(result)
                        else:
                            result = str(result)
                        print(
                            f"✅ Tool {tool_name} executed successfully, result length: {len(result)}"
                        )
                    else:
                        result = f"Tool {tool_name} does not have execute method"
                        print(f"❌ Tool {tool_name} missing execute method")
                else:
                    print(
                        f"⚠️ Tool {tool_name} not found in tools dict, using fallback methods..."
                    )
                    # Fallback to built-in methods
                    if tool_name == "search_file_knowledge":
                        search_result = await self.knowledge_search_method(
                            parameters.get("query", "")
                        )
                        if hasattr(search_result, "llm_formatted_context"):
                            result = search_result.llm_formatted_context or str(
                                search_result
                            )
                        else:
                            result = str(search_result)
                    elif tool_name == "vector_search":
                        search_result = await self.file_search_method(
                            parameters.get("query", "")
                        )
                        result = search_result.get(
                            "llm_formatted_context", str(search_result)
                        )
                    elif tool_name == "keyword_search":
                        search_result = await self.file_search_method(
                            parameters.get("query", "")
                        )
                        result = search_result.get(
                            "llm_formatted_context", str(search_result)
                        )
                    elif tool_name == "graph_traversal":
                        search_result = await self._call_retrieval_api(
                            query=parameters.get("query", ""), top_k_kg=15
                        )
                        result = search_result.get(
                            "llm_formatted_context", str(search_result)
                        )
                    else:
                        result = f"Unknown tool: {tool_name}"
                        print(f"❌ Unknown tool: {tool_name}")

                results.append(result)
                print(f"📝 Tool {tool_name} result preview: {result[:200]}...")

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                results.append(error_msg)
                print(f"❌ {error_msg}")

        print(
            f"🎯 Completed execution of {len(tool_calls)} tools, got {len(results)} results"
        )
        return results

    def _combine_iteration_results(self) -> str:
        """Combine results from multiple iterations."""
        combined = []
        for iteration, results in self.iteration_results.items():
            if isinstance(results, str):
                combined.append(f"Iteration {iteration}:\n{results}")
            else:
                combined.append(f"Iteration {iteration}:\n{str(results)}")

        return "\n\n".join(combined)

    async def arun(
        self, query: str, agent_config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        current_config = {**self.config, **(agent_config_override or {})}
        query_type = await self._classify_query_type(query)

        if "top_k_chunks" in current_config:
            dynamic_top_k = current_config.get("top_k_chunks")
        else:
            dynamic_top_k = await self._get_dynamic_top_k(query, query_type)

        initial_retrieval_num_sq = current_config.get("initial_retrieval_subqueries", 2)
        initial_retrieval_top_k_chunks = dynamic_top_k
        initial_retrieval_top_k_kg = current_config.get(
            "initial_retrieval_top_k_kg", 10
        )

        messages: List[Dict[str, Any]] = []
        executed_tools: List[Dict[str, Any]] = []
        self.iteration_results = {}

        initial_context = "No initial search performed."
        if current_config.get("perform_initial_retrieval", True):
            try:
                initial_search_results_dict = await self._call_retrieval_api(
                    query=query,
                    num_subqueries=initial_retrieval_num_sq,
                    top_k_chunks=initial_retrieval_top_k_chunks,
                    top_k_kg=initial_retrieval_top_k_kg,
                )
                initial_context = initial_search_results_dict.get(
                    "llm_formatted_context", "No initial search results found."
                )
                self.iteration_results[1] = initial_context
            except Exception as e:
                initial_context = f"Error during initial search: {str(e)}"
                self.iteration_results[1] = initial_context
        else:
            # Pure agentic mode - no initial retrieval, agent must use tools
            initial_context = "Pure agentic mode: No initial retrieval performed. Agent will use tools to gather all information."
            self.iteration_results[0] = (
                "Pure agentic mode activated - tools will handle all information retrieval"
            )
            print(
                "🎯 Pure agentic mode: No initial retrieval, agent will use tools for ALL information gathering"
            )

        current_date_str = current_config.get("current_date", "today")
        system_prompt_from_template = self.system_prompt_template.format(
            date=current_date_str
        )

        # CORRECTED: Removed the re-definitions of class methods from within arun.
        # This was inefficient and prone to bugs. The code now correctly calls the class methods like `self._enhanced_sufficiency_check`.

        # Check if this is pure agentic mode (no initial retrieval)
        is_pure_agentic_mode = not current_config.get("perform_initial_retrieval", True)

        if is_pure_agentic_mode:
            # In pure agentic mode, always use tools - never skip to direct answer
            sufficiency_assessment = {
                "is_sufficient": False,
                "reasoning": "Pure agentic mode: Agent must use tools for all information gathering.",
                "confidence": 1.0,
                "missing_information": ["All information must be gathered via tools"],
                "final_decision": False,
            }
            initial_sufficient = False
            print(
                "🎯 Pure agentic mode: Forcing tool usage for complete information gathering"
            )
        else:
            # Normal mode with initial retrieval - check sufficiency using GPT-4o-mini
            sufficiency_assessment = await self._enhanced_sufficiency_check(
                initial_context, query, query_type
            )
            initial_sufficient = sufficiency_assessment.get("final_decision", False)

            print(f"🤖 GPT-4o-mini sufficiency assessment:")
            print(f"   - Sufficient: {initial_sufficient}")
            print(
                f"   - Reasoning: {sufficiency_assessment.get('reasoning', 'No reasoning')}"
            )
            print(
                f"   - Confidence: {sufficiency_assessment.get('confidence', 0.0):.2f}"
            )

        self.iteration_results["sufficiency_assessment"] = sufficiency_assessment

        # Determine optimal tools early so we can use them in prompts
        optimal_tool_names = await self._determine_optimal_tools(query, query_type)

        if initial_sufficient:

            # Generate final answer directly from initial context
            final_answer_prompt = f"""
Based on the provided context, please provide a comprehensive and well-structured answer to the user's query.

**Query:** {query}
**Query Type:** {query_type}

**Context:**
{initial_context}

**Instructions:**
- Provide a direct, clear answer based on the context provided
- Structure your response appropriately for the query type
- Include relevant details and citations where appropriate
- Be concise but comprehensive

**Answer:**
"""

            try:
                llm_response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": final_answer_prompt}],
                    temperature=current_config.get("temperature", 0.3),
                    max_tokens=current_config.get("max_tokens_llm_response", 16000),
                )

                final_answer = llm_response.choices[0].message.content.strip()

                return {
                    "answer": final_answer,
                    "history": [
                        {
                            "role": "system",
                            "content": "Initial retrieval deemed sufficient by AI critique.",
                        },
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": final_answer},
                    ],
                    "tools_used": [],
                    "query_type": query_type,
                    "dynamic_top_k": dynamic_top_k,
                    "iterations_completed": 0,
                    "iteration_results": self.iteration_results,
                    "sufficiency_assessment": sufficiency_assessment,
                    "skipped_iterations": True,
                }

            except Exception as e:
                print(f"Error generating direct answer: {e}")
                # Fall back to iteration approach if direct answer fails
                pass

        # If initial results are insufficient or direct answer generation failed, proceed with iterations

        # Create instructional guidance based on mode
        if is_pure_agentic_mode:
            # Pure agentic mode instructions
            instructional_guidance = f"""
    **Pure Agentic Research Protocol (Query Type: {query_type}):**
    You are operating in PURE AGENTIC MODE where you must use tools to gather ALL information.

    **CRITICAL REQUIREMENT:** You MUST use the available tools to search for information. Do NOT attempt to answer the question without first using tools.

    **Mode:** Complete Tool-Based Research
    **Requirement:** You MUST use tools to gather information - no initial context is provided.

    **MANDATORY Research Process:**
    1. **Query Analysis:** Analyze the user's query: "{query}"
    2. **Tool Selection:** Choose the most appropriate tools from available options: {optimal_tool_names}
    3. **Information Gathering:** Execute selected tools to collect comprehensive information
    4. **Synthesis:** After gathering tool results, combine them into a complete, well-structured answer

    **IMPORTANT:** Start your response by using the function_calls format to execute the recommended tools.
    **Available Tools:** You have access to specialized search tools for different types of information retrieval.
    **Expectation:** Provide thorough, tool-based research with comprehensive coverage.

    **Example of how to start:**
    <function_calls>
    <invoke>
    <tool_name>search_file_knowledge</tool_name>
    <parameters>
    {{"query": "relevant search query based on user question"}}
    </parameters>
    </invoke>
    </function_calls>
    """
        else:
            # Normal mode with initial retrieval
            protocol_type = (
                "Single Iteration" if initial_sufficient else "Multi-Iteration"
            )
            initial_assessment = (
                "The initial retrieval has been evaluated by an AI critic and deemed sufficient for your query type: "
                + query_type
                + "."
                if initial_sufficient
                else "An AI critic has evaluated the initial retrieval and determined it is insufficient for your query type: "
                + query_type
                + "."
            )
            missing_info_line = (
                "**Missing Information:** "
                + ", ".join(
                    sufficiency_assessment.get("missing_aspects", ["Multiple aspects"])
                )
                if not initial_sufficient
                else ""
            )
            verification_type = (
                "Optional Verification"
                if initial_sufficient
                else "Mandatory Research Process"
            )
            analysis_text = (
                "context contains adequate information"
                if initial_sufficient
                else "Initial Context has gaps identified by the AI critic"
            )
            tool_use_type = "Optional" if initial_sufficient else "Comprehensive"
            tool_use_note = "(Optional)" if initial_sufficient else "(Required)"
            tool_permission = "may" if initial_sufficient else "MUST"
            tool_purpose = (
                "verify specific details"
                if initial_sufficient
                else "gather the missing information"
            )
            synthesis_instruction = (
                "Provide your final answer based on the sufficient context provided"
                if initial_sufficient
                else "Combine all gathered information for a comprehensive response"
            )

            instructional_guidance = f"""
    **Research Protocol ({protocol_type} - Query Type: {query_type}):**
    {initial_assessment}

    **LLM Assessment:** {sufficiency_assessment.get('reasoning')}
    **Confidence Level:** {sufficiency_assessment.get('confidence', 0.0):.2f}
    {missing_info_line}

    **{verification_type}:**
    1. **Initial Analysis:** The provided {analysis_text}.
    2. **{tool_use_type} Tool Use {tool_use_note}:** You {tool_permission} use tools to {tool_purpose}.
    3. **Synthesize and Answer:** {synthesis_instruction}.
    """

        tool_prompt_part = self._construct_tool_prompt_part(
            list(self.tools.values()), optimal_tool_names
        )

        if is_pure_agentic_mode:
            # Pure agentic mode - no initial context section
            system_prompt = f"""{system_prompt_from_template}
{instructional_guidance}

{tool_prompt_part}

**CRITICAL INSTRUCTIONS FOR PURE AGENTIC MODE:**
- You are in PURE AGENTIC MODE with NO initial context
- You MUST use tools to gather ALL information needed to answer: "{query}"
- Analyze the query and select the most appropriate tools from: {optimal_tool_names}
- Do NOT provide any analysis or answer until you have used tools to gather information
- Your first response MUST contain function_calls to execute your selected tools

**REQUIRED FIRST ACTION:** Use function_calls format to execute tools for information gathering.

**EXAMPLE OF REQUIRED FORMAT:**
<function_calls>
<invoke>
<tool_name>search_file_knowledge</tool_name>
<parameters>
{{"query": "{query}"}}
</parameters>
</invoke>
<invoke>
<tool_name>vector_search</tool_name>
<parameters>
{{"query": "{query}"}}
</parameters>
</invoke>
</function_calls>

**YOU MUST START WITH THE ABOVE FORMAT - DO NOT WRITE ANYTHING ELSE FIRST!**"""
        else:
            # Normal mode with initial context
            system_prompt = f"""{system_prompt_from_template}
{instructional_guidance}
### Initial Context (Iteration 1)
{initial_context}

{tool_prompt_part}
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        iterations_count = 0
        current_max_iterations = current_config.get(
            "max_iterations", self.max_iterations
        )

        # If initial results are sufficient, limit to 1 iteration max
        if initial_sufficient:
            current_max_iterations = 1
            print(
                f"🎯 GPT-4o-mini determined initial context is sufficient. Limiting to 1 iteration."
            )
        else:
            print(
                f"🔄 GPT-4o-mini determined initial context is insufficient. Proceeding with up to {current_max_iterations} iterations."
            )

        while iterations_count < current_max_iterations:
            iterations_count += 1
            print(f"\n--- Starting Iteration {iterations_count} ---")

            try:
                llm_response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=current_config.get("temperature", 0.3),
                    max_tokens=current_config.get("max_tokens_llm_response", 16000),
                    stop=["</function_calls>"],
                )
            except Exception as e:
                return {
                    "answer": "Error: LLM API call failed.",
                    "history": messages,
                    "error": str(e),
                    "tools_used": executed_tools,
                    "query_type": query_type,
                    "dynamic_top_k": dynamic_top_k,
                }

            assistant_message = llm_response.choices[0].message
            response_content = (
                assistant_message.content if assistant_message.content else ""
            )
            finish_reason = llm_response.choices[0].finish_reason

            if finish_reason == "stop":
                response_content += "</function_calls>"

            current_assistant_response_message = {
                "role": "assistant",
                "content": response_content,
            }

            parsed_tool_calls = self._parse_llm_tool_calls(response_content)

            # Handle case where LLM didn't use function calls in pure agentic mode
            if not parsed_tool_calls and is_pure_agentic_mode and iterations_count == 1:
                print(
                    f"🚨 Pure agentic mode: No tool calls detected on first iteration, prompting LLM to use tools..."
                )

                # Instead of forcing all tools, prompt the LLM to make tool selections
                tool_selection_prompt = f"""
You are in PURE AGENTIC MODE and MUST use tools to gather information. You did not use any tools in your previous response.

**CRITICAL REQUIREMENT:** You MUST use function_calls to execute tools before providing any answer.

**Available Tools:** {optimal_tool_names}

**Your Query:** {query}

**Instructions:**
1. Analyze the query and determine which 1-3 tools would be most effective
2. Use function_calls format to execute the selected tools
3. Do NOT provide an answer until you have tool results

**Example Format:**
<function_calls>
<invoke>
<tool_name>search_file_knowledge</tool_name>
<parameters>
{{"query": "your refined search query"}}
</parameters>
</invoke>
</function_calls>

Please select and execute the most appropriate tools for this query now:"""

                messages.append(current_assistant_response_message)
                messages.append({"role": "user", "content": tool_selection_prompt})
                continue  # Continue to next iteration to let LLM select tools

            if parsed_tool_calls:
                print(
                    f"Iteration {iterations_count}: Executing {len(parsed_tool_calls)} tool calls"
                )
                messages.append(current_assistant_response_message)

                # Track executed tools
                for tool_call in parsed_tool_calls:
                    executed_tools.append(
                        {
                            "tool_name": tool_call["tool_name"],
                            "parameters": tool_call["parameters"],
                            "iteration": iterations_count,
                        }
                    )

                # NEW: Execute tool calls in parallel
                tool_results = await self._execute_tool_calls_in_parallel(
                    parsed_tool_calls, current_config, messages
                )

                # Store tool results for this iteration
                iteration_tool_results = []
                all_results_content = []
                for i, tool_call in enumerate(parsed_tool_calls):
                    tool_name = tool_call["tool_name"]
                    tool_result_str = tool_results[i]
                    iteration_tool_results.append(
                        f"Tool: {tool_name}\nResult: {tool_result_str}"
                    )

                    result_block = f"""<result>
<tool_name>{tool_name}</tool_name>
<stdout>
{tool_result_str}
</stdout>
</result>"""
                    all_results_content.append(result_block)

                # Store results from this iteration
                if iterations_count not in self.iteration_results:
                    self.iteration_results[iterations_count] = ""
                self.iteration_results[iterations_count] += "\n".join(
                    iteration_tool_results
                )

                aggregated_tool_results_message = f"<function_results>\n{''.join(all_results_content)}\n</function_results>"

                messages.append(
                    {"role": "user", "content": aggregated_tool_results_message}
                )

                # After first iteration with tools, check if we should continue to second iteration
                if (
                    iterations_count == 1
                    and current_max_iterations > 1
                    and not is_pure_agentic_mode
                ):
                    # Combine current context with tool results for assessment
                    combined_context_for_assessment = (
                        initial_context + "\n\n" + "\n".join(iteration_tool_results)
                    )

                    # Use GPT-4o-mini to decide if second iteration is needed
                    second_iteration_assessment = (
                        await self._enhanced_sufficiency_check(
                            combined_context_for_assessment, query, query_type
                        )
                    )

                    is_now_sufficient = second_iteration_assessment.get(
                        "final_decision", False
                    )

                    print(f"🤖 Post-iteration 1 GPT-4o-mini assessment:")
                    print(f"   - Sufficient after iteration 1: {is_now_sufficient}")
                    print(
                        f"   - Reasoning: {second_iteration_assessment.get('reasoning', 'No reasoning')}"
                    )
                    print(
                        f"   - Confidence: {second_iteration_assessment.get('confidence', 0.0):.2f}"
                    )

                    if is_now_sufficient:
                        print(
                            f"🎯 GPT-4o-mini determined context is now sufficient after iteration 1. Skipping iteration 2."
                        )
                        current_max_iterations = 1  # Force exit after this iteration
                        self.iteration_results["second_iteration_assessment"] = (
                            second_iteration_assessment
                        )
                    else:
                        print(
                            f"🔄 GPT-4o-mini determined more information needed. Proceeding to iteration 2."
                        )
                        self.iteration_results["second_iteration_assessment"] = (
                            second_iteration_assessment
                        )
            else:
                # No more tool calls detected
                if (
                    is_pure_agentic_mode
                    and iterations_count == 1
                    and len(executed_tools) == 0
                ):
                    # In pure agentic mode, if no tools were executed at all, give one more chance for LLM to select tools
                    print(
                        f"🚨 Pure agentic mode: No tools executed yet, giving LLM one more chance to select tools..."
                    )

                    # Add a strong prompt to force tool usage
                    force_tool_prompt = f"""
**CRITICAL ERROR:** You are in Pure Agentic Mode but have not used any tools yet!

**MANDATORY REQUIREMENT:** You MUST use function_calls to execute tools before providing any answer.

**Query:** {query}
**Available Tools:** {optimal_tool_names}

**You MUST:**
1. Select 1-3 most appropriate tools from the available list
2. Use function_calls format to execute them
3. Wait for results before providing your final answer

**Example:**
<function_calls>
<invoke>
<tool_name>vector_search</tool_name>
<parameters>
{{"query": "refined search query based on user question"}}
</parameters>
</invoke>
</function_calls>

Execute the appropriate tools NOW:"""

                    messages.append({"role": "user", "content": force_tool_prompt})

                    # Try one more iteration to get tool usage
                    if iterations_count < current_max_iterations:
                        continue
                    else:
                        print(
                            f"⚠️ Max iterations reached without tool usage in pure agentic mode"
                        )

                # Generate final answer
                print(
                    f"Iteration {iterations_count}: No tool calls detected, generating final answer"
                )

                # Combine all iteration results for final context
                if len(self.iteration_results) > 1:
                    combined_context = self._combine_iteration_results()

                    # Update system prompt with combined results
                    final_system_prompt = f"""{system_prompt_from_template}

### Combined Results from All Iterations
{combined_context}

**Final Synthesis Instructions:**
Based on the information gathered across multiple iterations, provide a comprehensive, well-structured answer.
Clearly indicate which information came from which iteration if relevant.
Query Type: {query_type}
Dynamic Top-K Used: {dynamic_top_k}
"""
                    # Update the system message
                    messages[0]["content"] = final_system_prompt

                final_answer = (
                    re.sub(
                        r"<function_calls>.*</function_calls>",
                        "",
                        response_content,
                        flags=re.DOTALL,
                    )
                    .replace("</function_calls>", "")
                    .strip()
                )
                messages.append({"role": "assistant", "content": final_answer})

                return {
                    "answer": final_answer,
                    "history": messages,
                    "tools_used": executed_tools,
                    "query_type": query_type,
                    "dynamic_top_k": dynamic_top_k,
                    "iterations_completed": iterations_count,
                    "iteration_results": self.iteration_results,
                    "sufficiency_assessment": sufficiency_assessment,
                    "skipped_iterations": iterations_count
                    < current_config.get("max_iterations", self.max_iterations),
                }

        # Max iterations reached
        last_llm_content = (
            messages[-1]["content"]
            if messages and messages[-1]["role"] == "assistant"
            else "Max iterations reached without a conclusive answer."
        )

        # Combine results if we have multiple iterations
        if len(self.iteration_results) > 1:
            combined_context = self._combine_iteration_results()
            last_llm_content += f"\n\nCombined Research Results:\n{combined_context}"

        return {
            "answer": last_llm_content,
            "history": messages,
            "warning": "Max iterations reached",
            "tools_used": executed_tools,
            "query_type": query_type,
            "dynamic_top_k": dynamic_top_k,
            "iterations_completed": iterations_count,
            "iteration_results": self.iteration_results,
            "sufficiency_assessment": sufficiency_assessment,
            "skipped_iterations": False,
        }

    # NEW: Replace direct retriever calls with API calls - FIXED for proper agentic behavior
    async def _call_retrieval_api(
        self,
        query: str,
        num_subqueries: int = 2,
        top_k_chunks: int = 20,
        top_k_kg: int = 10,
        graph_query: str = None,
    ) -> Dict[str, Any]:
        """
        This method should only be used for initial retrieval, not for tool-based searches.
        For agentic behavior, tools should call their specific search methods.
        """
        if self.retriever:
            # Use retriever directly for initial retrieval only
            try:
                results = await self.retriever.search(
                    user_query=query,
                    num_subqueries=num_subqueries,
                    initial_candidate_pool_size=top_k_chunks,
                    top_k_kg_entities=top_k_kg,
                )
                return {
                    "llm_formatted_context": results.get("llm_formatted_context", ""),
                    "sub_queries_results": results.get("sub_queries_results", []),
                }
            except Exception as e:
                print(f"Direct retriever error: {e}")
                return {
                    "llm_formatted_context": f"Retriever error: {e}",
                    "sub_queries_results": [],
                }
        else:
            # Fallback to API call if no direct retriever
            try:
                timeout = httpx.Timeout(30.0, connect=10.0)
                # Create and properly close httpx client
                async with httpx.AsyncClient(timeout=timeout) as client:
                    payload = {
                        "question": query,
                        "index_name": self.config.get("index_name", "default"),
                        "top_k_chunks": top_k_chunks,
                        "enable_references_citations": True,
                        "deep_research": False,
                        "auto_chunk_sizing": True,
                        "model": self.config.get("model", "gpt-4o-mini"),
                    }

                    response = await client.post(
                        f"{API_BASE_URL}/query-rag",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.config.get('api_token', '')}"
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return {
                            "llm_formatted_context": result.get("answer", ""),
                            "sub_queries_results": result.get("context", []),
                        }
                    else:
                        return {
                            "llm_formatted_context": f"API error: {response.status_code}",
                            "sub_queries_results": [],
                        }

            except Exception as e:
                print(f"API call error: {e}")
                return {
                    "llm_formatted_context": f"API error: {e}",
                    "sub_queries_results": [],
                }

    # Enhanced API call with exponential backoff
    async def _call_retrieval_api_with_backoff(
        self,
        query: str,
        num_subqueries: int = 2,
        top_k_chunks: int = 20,
        top_k_kg: int = 10,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Call the retrieval API with exponential backoff for better reliability."""
        for attempt in range(max_retries):
            try:
                result = await self._call_retrieval_api(
                    query, num_subqueries, top_k_chunks, top_k_kg
                )
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) * 0.5  # 0.5, 1, 2 seconds
                    print(f"Retrieval API error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    # Return minimal valid structure
                    return {"llm_formatted_context": "", "sub_queries_results": []}

    async def knowledge_search_method(
        self, query: str, agent_config: Optional[Dict[str, Any]] = None
    ) -> AggregateSearchResult:
        effective_config = agent_config or self.config

        # Use API instead of direct retriever
        search_result = await self._call_retrieval_api(
            query=query,
            num_subqueries=effective_config.get("tool_num_subqueries", 1),
            top_k_chunks=effective_config.get("tool_top_k_chunks", 10),
            top_k_kg=effective_config.get("tool_top_k_kg", 10),
        )

        return AggregateSearchResult(
            query=query,
            llm_formatted_context=search_result.get("llm_formatted_context"),
        )

    async def file_search_method(
        self, query: str, agent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        effective_config = agent_config or self.config
        return await self._call_retrieval_api(
            query=query,
            num_subqueries=effective_config.get("tool_file_search_subqueries", 1),
            top_k_chunks=effective_config.get("tool_file_search_top_k_chunks", 10),
            top_k_kg=0,
        )

    async def content_method(
        self,
        filters: Dict,
        agent_config: Optional[Dict[str, Any]] = None,
        options: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        effective_config = agent_config or self.config
        doc_id_filter = filters.get("id", {}).get("$eq")
        if doc_id_filter:
            query = f"Retrieve all content for document ID {doc_id_filter}"
            # CHANGE: Use API instead of direct retriever
            return await self._call_retrieval_api(
                query=query,
                num_subqueries=1,
                top_k_chunks=effective_config.get("tool_content_top_k_chunks", 10),
                top_k_kg=0,
            )
        return {
            "error": "Document ID filter not correctly processed or missing in agent's content_method",
            "sub_queries_results": [],
        }

    async def _determine_optimal_tools(self, query: str, query_type: str) -> List[str]:
        """
        Uses GPT-4o-mini to intelligently determine the optimal set of tools for a given query.
        Returns a list of tool names to use based on LLM analysis of query requirements.
        """
        available_tools = {
            "vector_search": "Semantic similarity search using embeddings - best for conceptual queries and finding related content",
            "keyword_search": "Exact phrase matching - best for finding specific terms, names, or precise quotes",
            "search_file_knowledge": "Comprehensive document analysis - best for summaries, overviews, and broad information gathering",
            "graph_traversal": "Entity relationship analysis - best for queries about connections, relationships, and entity interactions",
        }

        tool_selection_prompt = f"""
You are an expert tool selection system. Analyze the user query and determine which tools would be most effective.

Available Tools:
{json.dumps(available_tools, indent=2)}

Query Type: {query_type}
User Query: "{query}"

Instructions:
1. Analyze what the query is asking for
2. Consider the query type and complexity
3. Select 1-3 most relevant tools (avoid selecting all tools unless truly necessary)
4. Prioritize quality over quantity - choose tools that best match the query intent
5. Consider efficiency - fewer, well-chosen tools often work better than many tools

Return ONLY a JSON array of tool names, like: ["vector_search", "keyword_search", "search_file_knowledge", "graph_traversal"]

Selected tools:"""

        try:
            response = await self._call_llm_api(
                tool_selection_prompt, model="gpt-4o-mini", max_tokens=100
            )

            # Parse the JSON response
            try:
                # Extract JSON array from response
                json_start = response.find("[")
                json_end = response.rfind("]") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    selected_tools = json.loads(json_str)

                    # Validate tools exist
                    valid_tools = [
                        tool for tool in selected_tools if tool in available_tools
                    ]

                    if valid_tools:
                        print(
                            f"LLM selected {len(valid_tools)} tools for query type '{query_type}': {valid_tools}"
                        )
                        return valid_tools
                    else:
                        print(
                            "LLM returned invalid tools, falling back to heuristic selection"
                        )

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse LLM tool selection response: {e}")

        except Exception as e:
            print(f"Error in LLM tool selection: {e}")

        # Fallback to comprehensive tool selection for better results
        print("Using fallback comprehensive tool selection")

        query_lower = query.lower()

        # For complex analysis, use comprehensive but selective approach
        if query_type == "complex_analysis":
            fallback_tools = [
                "search_file_knowledge",
                "vector_search",
                "keyword_search",
            ]
        elif query_type == "comparison":
            fallback_tools = [
                "search_file_knowledge",
                "vector_search",
                "keyword_search",
            ]
        elif query_type == "summary_extraction":
            fallback_tools = ["search_file_knowledge", "vector_search"]
        elif query_type == "factual_lookup":
            if any(term in query_lower for term in ["exact", "specifically", "quote"]):
                fallback_tools = ["keyword_search", "vector_search"]
            else:
                fallback_tools = ["vector_search", "search_file_knowledge"]
        else:
            # Default to comprehensive search
            fallback_tools = [
                "search_file_knowledge",
                "vector_search",
                "keyword_search",
            ]

        print(f"Fallback selected {len(fallback_tools)} tools: {fallback_tools}")
        return fallback_tools

    def _construct_tool_prompt_part(
        self, tools: List[Any], optimal_tool_names: Optional[List[str]] = None
    ) -> str:
        # Use provided optimal tool names or fall back to all tools
        if optimal_tool_names is None:
            optimal_tool_names = [tool.name for tool in tools]

        # Filter tools list to only include optimal tools
        optimal_tools = [tool for tool in tools if tool.name in optimal_tool_names]

        def format_params(parameters):
            if not parameters or "properties" not in parameters:
                return ""
            props = parameters["properties"]
            return "\n".join(
                [
                    f'<parameter><name>{name}</name><type>{prop.get("type", "any")}</type><description>{prop.get("description", "")}</description></parameter>'
                    for name, prop in props.items()
                ]
            )

        tool_strings = []
        for tool in optimal_tools:
            param_str = format_params(tool.parameters)
            tool_str = f"""<tool_description>
<tool_name>{tool.name}</tool_name>
<description>{tool.description}</description>
<parameters>
{param_str}
</parameters>
</tool_description>"""
            tool_strings.append(tool_str)

        joined_tool_strings = "\n".join(tool_strings)

        return f"""### Tool Usage
In this environment you have access to a set of tools you can use to answer the user's question.
You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
{{"$PARAMETER_NAME": "$PARAMETER_VALUE", ...}}
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
{joined_tool_strings}
</tools>"""


class GraphTraversalTool:
    """Tool for graph traversals and querying graph databases."""

    name = "graph_traversal"
    description = "Generate and execute graph database queries for relationship-oriented questions"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or query to analyze for entity relationships",
            }
        },
        "required": ["query"],
    }

    def __init__(self):
        self.context = None

    def set_context(self, context):
        self.context = context

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> str:
        if not self.context:
            return "Error: Graph traversal tool has no context."

        # Generate Cypher-like query from natural language
        cypher_query = await self._generate_graph_query(query)

        # Execute the query against the API
        result = await self._execute_graph_query(cypher_query, query)
        return result

    async def _generate_graph_query(self, query: str) -> str:
        """Generate a Cypher-like query from natural language using LLM."""
        if not self.context or not hasattr(self.context, "llm_client"):
            return ""

        # Create prompt for graph query generation
        prompt = f"""
        Convert this natural language query into a Cypher-style graph query:

        QUERY: {query}

        Focus on:
        - Identifying entities mentioned in the query
        - Determining relationships between entities
        - Expressing traversal paths between entities
        - Adding appropriate filtering conditions

        IMPORTANT: Return ONLY the Cypher-style query, no explanations.
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.context.llm_client.chat.completions.create(
                model=self.context.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating graph query: {e}")
            return ""

    async def _execute_graph_query(self, cypher_query: str, original_query: str) -> str:
        """Execute the generated query through the retrieval API."""
        if not self.context:
            return "Error: Graph traversal context not set."

        try:
            # Call API with generated query and original query
            response = await self.context._call_retrieval_api(
                query=original_query,
                graph_query=cypher_query,  # Pass the Cypher query to API
                num_subqueries=1,
                top_k_chunks=10,
                top_k_kg=15,  # Use higher KG retrieval for graph queries
            )
            return response.get("llm_formatted_context", "No graph results found.")
        except Exception as e:
            return f"Error executing graph query: {str(e)}"


async def main_research_agent_example():
    agent = None
    try:
        agent = StaticResearchAgent()
        query = input("Enter your research query for the StaticResearchAgent: ").strip()
        if not query:
            print("No query entered. Exiting.")
            return

        print(f"\n--- Running StaticResearchAgent for query: '{query}' ---")
        result = await agent.arun(query)

        print("\n--- Agent's Final Answer ---")
        print(result.get("answer", "No answer provided."))

        print(f"\n--- Query Analysis ---")
        print(f"Query Type: {result.get('query_type', 'Unknown')}")
        print(f"Dynamic Top-K: {result.get('dynamic_top_k', 'Unknown')}")
        print(f"Iterations Completed: {result.get('iterations_completed', 'Unknown')}")

        if result.get("tools_used"):
            print("\n--- Tools Used During Research ---")
            for i, tool_call in enumerate(result["tools_used"], 1):
                print(
                    f"{i}. {tool_call['tool_name']}({json.dumps(tool_call['parameters'])})"
                )

        if result.get("iteration_results"):
            print("\n--- Iteration Results Summary ---")
            for iteration, results in result["iteration_results"].items():
                print(f"Iteration {iteration}: {len(results)} characters of results")

        if result.get("warning"):
            print(f"\nWarning: {result['warning']}")
        if result.get("error"):
            print(f"\nError: {result['error']}")

    except ValueError as ve:
        print(f"Initialization error for StaticResearchAgent: {ve}")
    except Exception as e:
        print(f"An error occurred during the agent example run: {e}")
    finally:
        if agent:
            try:
                await agent.cleanup()
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    print("Running StaticResearchAgent example...")
    print("Please ensure your environment variables (e.g., OPENAI_API_KEY) are set.")
    asyncio.run(main_research_agent_example())
