import asyncio
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

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
except ImportError as e:
    print(
        f"ImportError in static_research_agent.py: {e}. Please ensure all dependencies are correctly placed and __init__.py files exist."
    )
    raise

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

DEFAULT_LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL")
DEFAULT_MAX_ITERATIONS = 2  # Changed from 5 to 2
DEFAULT_AGENT_CONFIG_PATH = (
    Path(__file__).parent.parent / "prompts" / "static_research_agent.yaml"
)


class StaticResearchAgent:
    def __init__(
        self,
        llm_client: Optional[AsyncOpenAI] = None,
        retriever: Optional[RAGFusionRetriever] = None,
        config_path: Optional[Union[str, Path]] = None,
        llm_model: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ):
        if llm_client is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found and llm_client not provided."
                )
            self.llm_client = AsyncOpenAI(api_key=openai_api_key)
        else:
            self.llm_client = llm_client

        if retriever is None:
            self.retriever = RAGFusionRetriever()
        else:
            self.retriever = retriever

        self.config_path = (
            Path(config_path) if config_path else DEFAULT_AGENT_CONFIG_PATH
        )
        self.config = self._load_config()

        self.llm_model = llm_model or self.config.get("llm_model", DEFAULT_LLM_MODEL)
        self.max_iterations = max_iterations or self.config.get(
            "max_iterations", DEFAULT_MAX_ITERATIONS
        )

        self.system_prompt_template = self._load_prompt_template_from_config(
            "static_research_agent"
        )

        # Initialize only the 3 imported tools with proper error handling
        self.tools = {}

        try:
            self.tools["search_file_knowledge"] = SearchFileKnowledgeTool()
        except Exception as e:
            print(f"Warning: Could not initialize SearchFileKnowledgeTool: {e}")

        try:
            self.tools["vector_search"] = VectorSearchTool()
        except Exception as e:
            print(f"Warning: Could not initialize VectorSearchTool: {e}")

        try:
            self.tools["keyword_search"] = KeywordSearchTool()
        except Exception as e:
            print(f"Warning: Could not initialize KeywordSearchTool: {e}")

        # Set context for tools if they need it
        for tool_instance in self.tools.values():
            if hasattr(tool_instance, "set_context"):
                tool_instance.set_context(self)

        # NEW: Store iteration results for combining
        self.iteration_results = {}

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
            try:
                prompt_file_path = (
                    Path(__file__).parent.parent / "prompts" / f"{prompt_key}.yaml"
                )
                with open(prompt_file_path, "r") as f:
                    data = yaml.safe_load(f)
                if data and prompt_key in data and "template" in data[prompt_key]:
                    return data[prompt_key]["template"]
                else:
                    raise ValueError(
                        f"Invalid or missing prompt structure for {prompt_key}"
                    )
            except Exception:
                return "You are a helpful assistant. Answer the user's query based on the provided context. Today's date is {date}."

    # NEW: Add query classification method to the agent
    async def _classify_query_type(self, query: str) -> str:
        """
        Uses an LLM to classify the user's query into a specific category.
        """
        classification_prompt = f"""
        You are an expert query classifier. Your task is to classify the user's query into one of the following categories based on its intent. Respond with ONLY the category name.

        Categories:
        - `factual_lookup`: For queries asking for specific, discrete pieces of information, like a number, name, date, or a specific definition. Example: "What is the lease start date for Tenant X?" or "How many floors does the building have?"
        - `summary_extraction`: For queries that ask for a summary of a topic, document, or concept. Example: "Summarize the main points of the contract." or "Provide an overview of the quarterly financial report."
        - `comparison`: For queries that ask to compare two or more items. Example: "Compare the lease terms for Tenant A and Tenant B."
        - `complex_analysis`: For broad, open-ended questions that require synthesizing information from multiple sources or analyzing relationships. This is the default for queries that don't fit other categories. Example: "What are the potential risks associated with the new project?"

        User Query: "{query}"

        Category:
        """
        valid_types = [
            "factual_lookup",
            "summary_extraction",
            "comparison",
            "complex_analysis",
        ]

        try:
            messages = [{"role": "user", "content": classification_prompt}]

            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=20,
                temperature=0.0,
            )

            query_type = response.choices[0].message.content.strip().lower()

            if query_type in valid_types:
                print(f"✅ Query classified as: '{query_type}'")
                return query_type
            else:
                print(
                    f"⚠️ Warning: LLM returned invalid query type '{query_type}'. Defaulting to 'complex_analysis'."
                )
                return "complex_analysis"

        except Exception as e:
            print(f"⚠️ Error classifying query: {e}. Defaulting to 'complex_analysis'.")
            return "complex_analysis"

    # NEW: Method to determine dynamic top_k based on query type
    def _get_dynamic_top_k(self, query_type: str) -> int:
        """
        Returns dynamic top_k chunks based on query classification.
        """
        if query_type == "factual_lookup":
            return 10
        else:
            return 20

    # NEW: Method to evaluate if initial results are sufficient
    def _are_initial_results_sufficient(self, initial_context: str, query: str) -> bool:
        """
        Evaluates if the initial retrieval results are sufficient to answer the query.
        Simple heuristic-based approach.
        """
        if (
            not initial_context
            or initial_context.strip() == "No initial search results found."
        ):
            return False

        # Simple heuristics to determine sufficiency
        context_length = len(initial_context.strip())

        # If context is too short, it's likely insufficient
        if context_length < 100:
            return False

        # Check if context contains key elements that suggest relevant information
        query_words = set(query.lower().split())
        context_words = set(initial_context.lower().split())
        overlap = len(query_words.intersection(context_words))

        # If there's minimal overlap, likely insufficient
        if overlap < 2:
            return False

        return True

    # NEW: Method to combine results from multiple iterations
    def _combine_iteration_results(self) -> str:
        """
        Combines results from multiple iterations with clear labeling.
        """
        combined_context = ""

        for iteration_num, result in self.iteration_results.items():
            combined_context += f"\n\n=== ITERATION {iteration_num} RESULTS ===\n"
            combined_context += result
            combined_context += f"\n=== END ITERATION {iteration_num} ===\n"

        return combined_context

    def _parse_llm_tool_calls(self, response_content: str) -> List[Dict[str, Any]]:
        tool_calls = []
        try:
            match = re.search(
                r"<function_calls>(.*?)</function_calls>", response_content, re.DOTALL
            )
            if not match:
                return []

            xml_str = f"<root>{match.group(1)}</root>"
            root = ET.fromstring(xml_str)

            for invoke_elem in root.findall("invoke"):
                name_elem = invoke_elem.find("tool_name")
                params_elem = invoke_elem.find("parameters")

                if name_elem is not None and name_elem.text:
                    tool_name = name_elem.text.strip()
                    parameters = {}
                    if (
                        params_elem is not None
                        and params_elem.text
                        and params_elem.text.strip()
                    ):
                        try:
                            parameters = json.loads(params_elem.text)
                        except json.JSONDecodeError:
                            continue
                    tool_calls.append(
                        {"tool_name": tool_name, "parameters": parameters}
                    )

        except ET.ParseError:
            pass
        except Exception:
            pass
        return tool_calls

    async def _execute_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        current_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> str:
        tool_instance = self.tools.get(tool_name)
        if not tool_instance:
            return f"Error: Tool '{tool_name}' not found."

        try:
            if hasattr(tool_instance, "execute"):
                return await tool_instance.execute(**parameters)

            elif tool_name == "search_file_knowledge":
                # Use the retriever's search functionality
                query = parameters.get("query", "")
                if not query:
                    return "Error: 'query' parameter is required for search_file_knowledge tool."

                search_result = await self.knowledge_search_method(
                    query, current_config
                )
                return search_result.llm_formatted_context or "No results found."

            elif tool_name == "vector_search":
                # Use the retriever's vector search
                query = parameters.get("query", "")
                if not query:
                    return (
                        "Error: 'query' parameter is required for vector_search tool."
                    )

                search_result = await self.retriever.search(
                    user_query=query,
                    num_subqueries=1,
                    initial_candidate_pool_size=current_config.get(
                        "tool_top_k_chunks", 5
                    ),
                    top_k_kg_entities=0,
                )
                return search_result.get("llm_formatted_context", "No results found.")

            elif tool_name == "keyword_search":
                # Use the retriever's keyword search functionality
                query = parameters.get("query", "")
                if not query:
                    return (
                        "Error: 'query' parameter is required for keyword_search tool."
                    )

                search_result = await self.file_search_method(query, current_config)
                return search_result.get("llm_formatted_context", "No results found.")

            else:
                return f"Error: Tool '{tool_name}' execution logic not implemented."

        except Exception as e:
            return f"Error: Failed to execute tool {tool_name}. Details: {str(e)}"

    async def arun(
        self, query: str, agent_config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        current_config = {**self.config, **(agent_config_override or {})}
        current_llm_model = current_config.get("llm_model", self.llm_model)
        current_max_iterations = current_config.get(
            "max_iterations", self.max_iterations
        )

        # NEW: Classify query type and set dynamic top_k
        query_type = await self._classify_query_type(query)
        dynamic_top_k = self._get_dynamic_top_k(query_type)

        print(
            f"Query classified as: {query_type}, using dynamic top_k: {dynamic_top_k}"
        )

        # Use dynamic top_k for initial retrieval
        initial_retrieval_num_sq = current_config.get("initial_retrieval_subqueries", 2)
        initial_retrieval_top_k_chunks = dynamic_top_k  # Use dynamic value
        initial_retrieval_top_k_kg = current_config.get(
            "initial_retrieval_top_k_kg", 10
        )

        messages: List[Dict[str, Any]] = []
        executed_tools: List[Dict[str, Any]] = []
        self.iteration_results = {}  # Reset iteration results

        # Initial Retrieval (Iteration 1)
        initial_context = "No initial search performed."
        if current_config.get("perform_initial_retrieval", True):
            try:
                initial_search_results_dict = await self.retriever.search(
                    user_query=query,
                    num_subqueries=initial_retrieval_num_sq,
                    top_k_chunks=initial_retrieval_top_k_chunks,
                    top_k_kg=initial_retrieval_top_k_kg,
                )
                initial_context = initial_search_results_dict.get(
                    "llm_formatted_context", "No initial search results found."
                )
                # Store iteration 1 results
                self.iteration_results[1] = initial_context
            except Exception as e:
                initial_context = f"Error during initial search: {str(e)}"
                self.iteration_results[1] = initial_context

        current_date_str = current_config.get("current_date", "today")
        system_prompt_from_template = self.system_prompt_template.format(
            date=current_date_str
        )

        # NEW: Check if initial results are sufficient
        initial_sufficient = self._are_initial_results_sufficient(
            initial_context, query
        )

        if initial_sufficient:
            print(
                "Initial results appear sufficient. Proceeding with single iteration."
            )
            instructional_guidance = f"""
**Research Protocol (Single Iteration - Query Type: {query_type}):**
The initial retrieval has provided sufficient information for your query type: {query_type}.

**Initial Context Analysis:** The provided 'Initial Context' contains relevant information for the query. However, you should still verify completeness and may use tools if you identify specific gaps.

**Instructions:**
1. **Analyze the Initial Context:** Review the provided context thoroughly.
2. **Optional Tool Use:** If you identify specific gaps or need additional verification, you may use tools:
   - Use `search_file_knowledge` to search through document contents and knowledge
   - Use `vector_search` for semantic similarity searches
   - Use `keyword_search` for exact keyword matching
3. **Synthesize and Answer:** Provide a comprehensive final answer based on all available information.
"""
        else:
            print(
                "Initial results appear insufficient. Will proceed with comprehensive tool-based research in iteration 2."
            )
            instructional_guidance = f"""
**Research Protocol (Two Iterations - Query Type: {query_type}):**
The initial retrieval may not provide complete information for your query type: {query_type}.

**Mandatory Research Process:**
1. **Initial Analysis:** The provided 'Initial Context' offers some clues but is likely insufficient for a complete answer.
2. **Comprehensive Tool Use (Required for Iteration 2):** You MUST use all available tools to gather comprehensive information:
   - Use `search_file_knowledge` to search through document contents and knowledge
   - Use `vector_search` for semantic similarity searches
   - Use `keyword_search` for exact keyword matching
   - Call multiple tools in parallel to be efficient
3. **Synthesize and Answer:** After tool execution, combine results from both the initial context and tool results to provide a comprehensive answer.

**Critical:** Given the query type '{query_type}', thorough research is essential for accuracy.
"""

        tool_prompt_part = self._construct_tool_prompt_part(list(self.tools.values()))

        system_prompt = f"""{system_prompt_from_template}
{instructional_guidance}
### Initial Context (Iteration 1)
{initial_context}

{tool_prompt_part}
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        iterations_count = 0
        while iterations_count < current_max_iterations:
            iterations_count += 1
            print(f"\n--- Starting Iteration {iterations_count} ---")

            try:
                llm_response = await self.llm_client.chat.completions.create(
                    model=current_llm_model,
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

            if parsed_tool_calls:
                print(
                    f"Iteration {iterations_count}: Executing {len(parsed_tool_calls)} tool calls"
                )
                messages.append(current_assistant_response_message)

                tasks = []
                for tool_call in parsed_tool_calls:
                    executed_tools.append(tool_call)
                    tool_name = tool_call["tool_name"]
                    tool_params = tool_call["parameters"]
                    # Use dynamic top_k for tool calls too
                    if "top_k_chunks" in current_config:
                        current_config["tool_top_k_chunks"] = dynamic_top_k
                    tasks.append(
                        self._execute_tool_call(
                            tool_name, tool_params, current_config, messages
                        )
                    )

                tool_results = await asyncio.gather(*tasks)

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
            else:
                # No more tool calls, generate final answer
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
        }

    async def knowledge_search_method(
        self, query: str, agent_config: Optional[Dict[str, Any]] = None
    ) -> AggregateSearchResult:
        effective_config = agent_config or self.config

        num_sq = effective_config.get("tool_num_subqueries", 1)
        top_k_c = effective_config.get("tool_top_k_chunks", 10)
        top_k_kg_val = effective_config.get("tool_top_k_kg", 10)

        raw_results_dict = await self.retriever.search(
            user_query=query,
            num_subqueries=num_sq,
            top_k_chunks=top_k_c,
            top_k_kg=top_k_kg_val,
        )
        llm_formatted_context = raw_results_dict.get("llm_formatted_context")
        agg_result = AggregateSearchResult(
            query=query, llm_formatted_context=llm_formatted_context
        )

        if raw_results_dict and "sub_queries_results" in raw_results_dict:
            for sq_res in raw_results_dict["sub_queries_results"]:
                for chunk_data in sq_res.get("reranked_chunks", []):
                    if not isinstance(chunk_data, dict):
                        continue
                    try:
                        agg_result.chunk_search_results.append(
                            ChunkSearchResult(
                                **{
                                    k: chunk_data.get(k)
                                    for k in ChunkSearchResult.__annotations__
                                }
                            )
                        )
                    except Exception:
                        pass

                for kg_data_item in sq_res.get("retrieved_kg_data", []):
                    if not isinstance(kg_data_item, dict):
                        continue
                    try:
                        entities = [
                            KGEntity(**e_data)
                            for e_data in kg_data_item.get("entities", [])
                            if isinstance(e_data, dict)
                        ]
                        relationships = [
                            KGRelationship(**r_data)
                            for r_data in kg_data_item.get("relationships", [])
                            if isinstance(r_data, dict)
                        ]
                        kg_search_result_data = {
                            **kg_data_item,
                            "entities": entities,
                            "relationships": relationships,
                        }
                        agg_result.graph_search_results.append(
                            KGSearchResult(
                                **{
                                    k: kg_search_result_data.get(k)
                                    for k in KGSearchResult.__annotations__
                                }
                            )
                        )
                    except Exception:
                        pass
        return agg_result

    async def file_search_method(
        self, query: str, agent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        effective_config = agent_config or self.config
        return await self.retriever.search(
            user_query=query,
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
            return await self.retriever.search(
                user_query=query,
                num_subqueries=1,
                top_k_chunks=effective_config.get("tool_content_top_k_chunks", 10),
                top_k_kg=0,
            )
        return {
            "error": "Document ID filter not correctly processed or missing in agent's content_method",
            "sub_queries_results": [],
        }

    def _construct_tool_prompt_part(self, tools: List[Any]) -> str:
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
        for tool in tools:
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
            if hasattr(agent, "llm_client") and isinstance(
                agent.llm_client, AsyncOpenAI
            ):
                if hasattr(agent.llm_client, "aclose"):
                    await agent.llm_client.aclose()
            if (
                hasattr(agent, "retriever")
                and hasattr(agent.retriever, "es_client")
                and agent.retriever.es_client
            ):
                if hasattr(agent.retriever.es_client, "close"):
                    await agent.retriever.es_client.close()


if __name__ == "__main__":
    print("Running StaticResearchAgent example...")
    print("Please ensure your environment variables (e.g., OPENAI_API_KEY) are set.")
    asyncio.run(main_research_agent_example())
