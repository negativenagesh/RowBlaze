import asyncio
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile
import subprocess
import sys

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from ..retrieval.new_rag_fusion import RAGFusionRetriever
    from ..tools.search_file_knowledge import SearchFileKnowledgeTool
    from ..tools.search_file_descriptions import SearchFileDescriptionsTool
    from ..tools.get_file_content import GetFileContentTool
    from ...utils.logging_config import setup_logger
    from ...core.base.abstractions import (
        AggregateSearchResult,
        ChunkSearchResult,
        KGSearchResult,
        KGEntity,
        KGRelationship,
    )
except ImportError as e:
    print(
        f"ImportError in static_research_agent.py: {e}. Please ensure all dependencies are correctly placed and __init__.py files exist."
    )
    print(
        "This agent expects RAGFusionRetriever, tools, and AggregateSearchResult to be importable from its location."
    )
    raise

load_dotenv()
logger = setup_logger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

DEFAULT_LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL")
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_AGENT_CONFIG_PATH = (
    Path(__file__).parent.parent / "prompts" / "static_research_agent.yaml"
)


class ReasoningTool:
    """Tool definition for the reasoning capability."""

    name = "reason"
    description = "Execute a reasoning query using a specialized reasoning LLM to analyze the conversation or a specific query. Use this for deep thinking, planning, and analysis."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The reasoning query to execute based on the conversation history.",
            }
        },
        "required": ["query"],
    }

    def set_context(self, context: Any):
        pass


class CritiqueTool:
    """Tool definition for the critique capability."""

    name = "critique"
    description = "Critique the conversation history to find logical fallacies, biases, or overlooked considerations. Helps improve the quality of the reasoning."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A specific question to guide the critique. Can be empty.",
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of areas to focus the critique on.",
            },
        },
        "required": [],
    }

    def set_context(self, context: Any):
        pass


class PythonInterpreterTool:
    """Tool definition for the Python interpreter."""

    name = "execute_python"
    description = "Executes Python code in a separate, isolated subprocess with a timeout. Use for calculations, data manipulation, simulations, etc."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute."},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 10).",
            },
        },
        "required": ["code"],
    }

    def set_context(self, context: Any):
        pass


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
            logger.info(
                "Initialized default AsyncOpenAI client for StaticResearchAgent."
            )
        else:
            self.llm_client = llm_client

        if retriever is None:
            self.retriever = RAGFusionRetriever()
            logger.info(
                "Initialized default RAGFusionRetriever for StaticResearchAgent."
            )
        else:
            self.retriever = retriever

        self.config_path = (
            Path(config_path) if config_path else DEFAULT_AGENT_CONFIG_PATH
        )
        self.config = self._load_config()

        self.llm_model = llm_model or self.config.get(DEFAULT_LLM_MODEL)
        self.max_iterations = max_iterations or self.config.get(DEFAULT_MAX_ITERATIONS)

        self.system_prompt_template = self._load_prompt_template_from_config(
            "static_research_agent"
        )

        # Initialize tools
        self.tools = {
            "search_file_knowledge": SearchFileKnowledgeTool(),
            "search_file_descriptions": SearchFileDescriptionsTool(),
            "get_file_content": GetFileContentTool(),
            "reason": ReasoningTool(),
            "critique": CritiqueTool(),
            "execute_python": PythonInterpreterTool(),
        }
        # Set context for tools if they need it (e.g., for calling agent's methods)
        for tool_instance in self.tools.values():
            if hasattr(tool_instance, "set_context"):
                tool_instance.set_context(self)
        logger.info(
            f"StaticResearchAgent initialized with model: {self.llm_model}, max_iterations: {self.max_iterations}"
        )
        logger.debug(f"Tools available: {list(self.tools.keys())}")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                logger.error(
                    f"Config file {self.config_path} did not load as a dictionary."
                )
                return {}
            logger.info(f"Successfully loaded agent config from {self.config_path}")
            return config_data
        except FileNotFoundError:
            logger.error(
                f"Agent config file not found: {self.config_path}. Using empty config."
            )
            return {}
        except Exception as e:
            logger.error(
                f"Error loading agent config from {self.config_path}: {e}",
                exc_info=True,
            )
            return {}

    def _load_prompt_template_from_config(self, prompt_key: str) -> str:
        prompt_details = self.config.get(prompt_key, {})
        if isinstance(prompt_details, dict) and "template" in prompt_details:
            logger.info(
                f"Successfully loaded prompt template for '{prompt_key}' from agent config."
            )
            return prompt_details["template"]
        else:
            logger.warning(
                f"Prompt template for '{prompt_key}' not found directly in agent config. Attempting to load from default prompts location."
            )
            try:
                prompt_file_path = (
                    Path(__file__).parent.parent / "prompts" / f"{prompt_key}.yaml"
                )
                with open(prompt_file_path, "r") as f:
                    data = yaml.safe_load(f)
                if data and prompt_key in data and "template" in data[prompt_key]:
                    logger.info(
                        f"Successfully loaded prompt template for '{prompt_key}' from {prompt_file_path}."
                    )
                    return data[prompt_key]["template"]
                else:
                    logger.error(
                        f"Prompt template for '{prompt_key}' not found or invalid in {prompt_file_path}."
                    )
                    raise ValueError(
                        f"Invalid or missing prompt structure for {prompt_key}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to load fallback prompt for {prompt_key}: {e}",
                    exc_info=True,
                )
                return "You are a helpful assistant. Answer the user's query based on the provided context. Today's date is {date}."

    def _parse_llm_tool_calls(self, response_content: str) -> List[Dict[str, Any]]:
        tool_calls = []
        try:
            # Match the user-specified <function_calls> block
            match = re.search(
                r"<function_calls>(.*?)</function_calls>", response_content, re.DOTALL
            )
            if not match:
                return []

            # The content inside <function_calls> is a series of <invoke> blocks
            # Wrap in a root element for valid XML parsing
            xml_str = f"<root>{match.group(1)}</root>"
            root = ET.fromstring(xml_str)

            for invoke_elem in root.findall("invoke"):
                name_elem = invoke_elem.find("tool_name")
                params_elem = invoke_elem.find("parameters")

                if name_elem is not None and name_elem.text:
                    tool_name = name_elem.text.strip()
                    parameters = {}
                    # Parameters are expected to be a single JSON blob inside the <parameters> tag
                    if (
                        params_elem is not None
                        and params_elem.text
                        and params_elem.text.strip()
                    ):
                        try:
                            parameters = json.loads(params_elem.text)
                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to parse JSON from <parameters> for tool {tool_name}. Content: {params_elem.text}"
                            )
                            continue  # Skip this malformed tool call
                    tool_calls.append(
                        {"tool_name": tool_name, "parameters": parameters}
                    )

        except ET.ParseError as e:
            logger.error(
                f"XML parsing error for tool calls: {e}. Content: {response_content[:500]}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error parsing tool calls: {e}. Content: {response_content[:500]}",
                exc_info=True,
            )
        return tool_calls

    async def _execute_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        current_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> str:
        logger.info(
            f"Attempting to execute tool: {tool_name} with parameters: {parameters}"
        )
        tool_instance = self.tools.get(tool_name)
        if not tool_instance:
            return f"Error: Tool '{tool_name}' not found."

        try:
            if tool_name == "search_file_knowledge":
                query = parameters.get("query")
                if not query:
                    return "Error: 'query' parameter missing for search_file_knowledge."
                tool_output_obj: AggregateSearchResult = (
                    await self.knowledge_search_method(
                        query=query, agent_config=current_config
                    )
                )
                return (
                    tool_output_obj.llm_formatted_context
                    or "No relevant information found by search_file_knowledge."
                )

            elif tool_name == "search_file_descriptions":
                query = parameters.get("query")
                if not query:
                    return (
                        "Error: 'query' parameter missing for search_file_descriptions."
                    )
                search_results_dict = await self.file_search_method(
                    query=query, agent_config=current_config
                )
                return search_results_dict.get(
                    "llm_formatted_context", "No relevant file descriptions found."
                )

            elif tool_name == "get_file_content":
                doc_id = parameters.get("document_id")
                if not doc_id:
                    return (
                        "Error: 'document_id' parameter missing for get_file_content."
                    )
                content_results_dict = await self.content_method(
                    filters={"id": {"$eq": doc_id}}, agent_config=current_config
                )
                return content_results_dict.get(
                    "llm_formatted_context",
                    f"Could not retrieve content for document ID {doc_id}.",
                )

            elif tool_name == "reason":
                query = parameters.get("query")
                if not query:
                    return "Error: 'query' parameter is required for reason tool."
                return await self._reason(query, messages, current_config)

            elif tool_name == "critique":
                return await self._critique(
                    query=parameters.get("query", ""),
                    focus_areas=parameters.get("focus_areas"),
                    messages=messages,
                    config=current_config,
                )

            elif tool_name == "execute_python":
                code = parameters.get("code")
                if not code:
                    return (
                        "Error: 'code' parameter is required for execute_python tool."
                    )
                timeout = parameters.get("timeout", 10)
                py_results = await self._execute_python_with_process_timeout(
                    code, timeout
                )
                return self._format_python_results(py_results)

            else:
                return f"Error: Tool '{tool_name}' execution logic not implemented."

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error: Failed to execute tool {tool_name}. Details: {str(e)}"

    async def arun(
        self, query: str, agent_config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        current_config = {**self.config, **(agent_config_override or {})}
        current_llm_model = current_config.get(self.llm_model)
        current_max_iterations = current_config.get(self.max_iterations)
        initial_retrieval_num_sq = current_config.get("initial_retrieval_subqueries", 2)
        initial_retrieval_top_k_chunks = current_config.get(
            "initial_retrieval_top_k_chunks", 3
        )
        initial_retrieval_top_k_kg = current_config.get(
            "initial_retrieval_top_k_kg", 10
        )

        logger.info(
            f"StaticResearchAgent starting 'arun' for query: \"{query}\" with model {current_llm_model}, max_iter={current_max_iterations}"
        )

        messages: List[Dict[str, Any]] = []
        executed_tools: List[Dict[str, Any]] = []

        # 1. Initial Retrieval (Optional, based on config)
        initial_context = "No initial search performed."  # Default
        if current_config.get("perform_initial_retrieval", True):
            try:
                logger.info("Performing initial retrieval...")
                initial_search_results_dict = await self.retriever.search(
                    user_query=query,
                    num_subqueries=initial_retrieval_num_sq,
                    top_k_chunks=initial_retrieval_top_k_chunks,
                    top_k_kg=initial_retrieval_top_k_kg,
                )
                initial_context = initial_search_results_dict.get(
                    "llm_formatted_context", "No initial search results found."
                )
                logger.debug(
                    f"Initial context formatted (first 500 chars): {initial_context[:500]}"
                )
            except Exception as e:
                logger.error(f"Error during initial retrieval: {e}", exc_info=True)
                initial_context = "Error during initial search."

        # 2. Prepare System Prompt
        current_date_str = current_config.get("current_date", "today")

        system_prompt_from_template = self.system_prompt_template.format(
            date=current_date_str
        )

        instructional_guidance = """
**Mandatory Research Protocol:**
You are a research agent. Your primary function is to use tools to gather information. You must follow this protocol for every query:

1.  **Initial Analysis:** The user query and the 'Initial Context' are provided. Treat the 'Initial Context' as a high-level summary or a set of clues, NOT as the final source of truth. It is **never** sufficient for a complete answer.

2.  **Mandatory Tool Use:** You **MUST** use your tools to dig deeper. Based on the clues in the initial context (like document IDs or key topics), formulate one or more tool calls.
    *   If you see document IDs, use `get_file_content` to read the full document.
    *   If you see interesting topics, use `search_file_knowledge` to find more details or related information.
    *   You can and should call multiple tools in parallel to be efficient.

3.  **Synthesize and Answer:** After you have received the results from your tool calls, and only then, synthesize all the information (initial context + tool results) into a comprehensive final answer.

**Crucial Rule:** Never provide a final answer based only on the initial context. Your value is in the deep research you perform using your tools. An answer without a preceding tool call in the conversation history is a failure to follow protocol.
"""

        tool_prompt_part = self._construct_tool_prompt_part(list(self.tools.values()))

        system_prompt = f"""{system_prompt_from_template}
{instructional_guidance}
### Initial Context
{initial_context}

{tool_prompt_part}
"""
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        iterations_count = 0
        while iterations_count < current_max_iterations:
            iterations_count += 1
            logger.info(f"Agent Iteration {iterations_count}/{current_max_iterations}")

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Messages to LLM:\n{json.dumps(messages, indent=2, default=str)}"
                )

            try:
                llm_response = await self.llm_client.chat.completions.create(
                    model=current_llm_model,
                    messages=messages,
                    temperature=current_config.get("temperature", 0.3),
                    max_tokens=current_config.get("max_tokens_llm_response", 16000),
                    stop=[
                        "</function_calls>"
                    ],  # Advise model to stop after calling functions
                )
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}", exc_info=True)
                return {
                    "answer": "Error: LLM API call failed.",
                    "history": messages,
                    "error": str(e),
                    "tools_used": executed_tools,
                }

            assistant_message = llm_response.choices[0].message
            response_content = (
                assistant_message.content if assistant_message.content else ""
            )
            finish_reason = llm_response.choices[0].finish_reason

            # Append the stop sequence back if the model used it
            if finish_reason == "stop":
                response_content += "</function_calls>"

            logger.debug(f"LLM Raw Response Content:\n{response_content}")
            logger.debug(f"LLM Finish Reason: {finish_reason}")

            current_assistant_response_message = {
                "role": "assistant",
                "content": response_content,
            }

            parsed_tool_calls = self._parse_llm_tool_calls(response_content)

            if parsed_tool_calls:
                logger.info(
                    f"LLM requested {len(parsed_tool_calls)} tool calls: {parsed_tool_calls}"
                )
                messages.append(current_assistant_response_message)

                # Create tasks for parallel execution
                tasks = []
                for tool_call in parsed_tool_calls:
                    executed_tools.append(tool_call)  # Track the tool call
                    tool_name = tool_call["tool_name"]
                    tool_params = tool_call["parameters"]
                    tasks.append(
                        self._execute_tool_call(
                            tool_name, tool_params, current_config, messages
                        )
                    )

                # Execute tools in parallel
                tool_results = await asyncio.gather(*tasks)

                # Aggregate results into a single message
                all_results_content = []
                for i, tool_call in enumerate(parsed_tool_calls):
                    tool_name = tool_call["tool_name"]
                    tool_result_str = tool_results[i]

                    result_block = f"""<result>
<tool_name>{tool_name}</tool_name>
<stdout>
{tool_result_str}
</stdout>
</result>"""
                    all_results_content.append(result_block)
                    logger.debug(
                        f"Gathered tool result for {tool_name}: {tool_result_str[:200]}..."
                    )

                aggregated_tool_results_message = f"<function_results>\n{''.join(all_results_content)}\n</function_results>"

                messages.append(
                    {"role": "user", "content": aggregated_tool_results_message}
                )
            else:
                # No tool calls, so this is the final answer.
                # Clean up any potential tool-related XML tags from the final response.
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
                logger.info("LLM provided final answer.")
                messages.append({"role": "assistant", "content": final_answer})
                return {
                    "answer": final_answer,
                    "history": messages,
                    "tools_used": executed_tools,
                }

        logger.warning(f"Max iterations ({current_max_iterations}) reached.")
        last_llm_content = (
            messages[-1]["content"]
            if messages and messages[-1]["role"] == "assistant"
            else "Max iterations reached without a conclusive answer."
        )
        return {
            "answer": last_llm_content,
            "history": messages,
            "warning": "Max iterations reached",
            "tools_used": executed_tools,
        }

    async def knowledge_search_method(
        self, query: str, agent_config: Optional[Dict[str, Any]] = None
    ) -> AggregateSearchResult:
        effective_config = (
            agent_config or self.config
        )  # Use passed config or agent's default
        logger.debug(f"Agent's knowledge_search_method called with query: {query}")

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
                    except Exception as e_chunk:
                        logger.error(
                            f"Error creating ChunkSearchResult from data: {chunk_data}, error: {e_chunk}",
                            exc_info=True,
                        )

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
                    except Exception as e_kg:
                        logger.error(
                            f"Error creating KGSearchResult from data: {kg_data_item}, error: {e_kg}",
                            exc_info=True,
                        )
        return agg_result

    async def file_search_method(
        self, query: str, agent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        effective_config = agent_config or self.config
        logger.debug(f"Agent's file_search_method called with query: {query}")
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
        logger.debug(f"Agent's content_method called with filters: {filters}")
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

        # Pre-join the tool strings to avoid putting a string with a backslash ('\n')
        # inside an f-string expression, which causes a SyntaxError in Python < 3.12.
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

    async def _reason(
        self, query: str, messages: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> str:
        """
        Executes a reasoning query using the agent's primary LLM to perform deep thinking,
        planning, or analysis on the conversation history.
        """
        logger.info(
            f"Executing reasoning tool with query: '{query}' using the agent's primary LLM."
        )

        # Construct a new set of messages for the reasoning task.
        reasoning_messages = messages + [{"role": "user", "content": query}]

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=reasoning_messages,
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens_llm_response", 16000),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Reasoning tool LLM call failed: {e}", exc_info=True)
            return f"Error during reasoning: {e}"

    async def _critique(
        self,
        query: str,
        focus_areas: Optional[List[str]],
        messages: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> str:
        """Critique the conversation history."""
        logger.info(
            f"Executing critique tool with query: {query} and focus areas: {focus_areas}"
        )
        if not focus_areas:
            focus_areas = []

        # Build the critique prompt
        critique_prompt = (
            "You are a critical reasoning expert. Your task is to analyze the following conversation "
            "and critique the reasoning. Look for:\n"
            "1. Logical fallacies or inconsistencies\n"
            "2. Cognitive biases\n"
            "3. Overlooked questions or considerations\n"
            "4. Alternative approaches\n"
            "5. Improvements in rigor\n\n"
        )

        if focus_areas:
            critique_prompt += f"Focus areas: {', '.join(focus_areas)}\n\n"

        if query.strip():
            critique_prompt += f"Specific question: {query}\n\n"

        critique_prompt += (
            "Structure your critique:\n"
            "1. Summary\n"
            "2. Key strengths\n"
            "3. Potential issues\n"
            "4. Alternatives\n"
            "5. Recommendations\n\n"
            "--- CONVERSATION HISTORY ---\n"
        )

        # Add the conversation history to the prompt
        conversation_text = "\n".join(
            f"{msg.get('role', '').upper()}: {msg.get('content', '')}"
            for msg in messages
            if msg.get("content") and msg.get("role") in ["user", "assistant", "system"]
        )

        final_prompt = critique_prompt + conversation_text

        # Use the simplified reason method to process the critique
        return await self._reason(final_prompt, messages, config)

    async def _execute_python_with_process_timeout(
        self, code: str, timeout: int = 10
    ) -> dict[str, Any]:
        """
        Executes Python code in a separate subprocess with a timeout.
        This provides isolation and prevents re-importing the current agent module.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(code)
            script_path = tmp_file.name
        try:
            # Run the script in a fresh subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "timed_out": False,
                "timeout": timeout,
                "error": (
                    None
                    if result.returncode == 0
                    else {
                        "type": "SubprocessError",
                        "message": f"Process exited with code {result.returncode}",
                    }
                ),
            }
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
                "success": False,
                "timed_out": True,
                "timeout": timeout,
                "error": {
                    "type": "TimeoutError",
                    "message": f"Execution exceeded {timeout} second limit.",
                },
            }
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def _format_python_results(self, results: dict[str, Any]) -> str:
        """Format Python execution results for display."""
        output = []

        if results.get("timed_out"):
            output.append(
                f"⚠️ **Execution Timeout**: Code exceeded the {results.get('timeout', 10)} second limit."
            )

        if results.get("stdout"):
            output.append(
                "## Output (stdout):\n```\n" + results["stdout"].rstrip() + "\n```"
            )

        if not results.get("success"):
            output.append("## Error (stderr):\n```")
            stderr_out = results.get("stderr", "").rstrip()
            if stderr_out:
                output.append(stderr_out)

            err_obj = results.get("error")
            if err_obj and err_obj.get("message"):
                if stderr_out:
                    output.append("\n")  # Add a newline for separation
                output.append(err_obj["message"])
            output.append("```")

        return (
            "\n".join(output)
            if output
            else "Code executed successfully with no output."
        )


async def main_research_agent_example():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    logging.getLogger("src.core.agents.static_research_agent").setLevel(logging.DEBUG)
    logging.getLogger("src.core.retrieval.new_rag_fusion").setLevel(logging.INFO)

    agent = None  # Initialize agent to None for finally block
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

        if result.get("tools_used"):
            print("\n--- Tools Used During Research ---")
            for i, tool_call in enumerate(result["tools_used"], 1):
                print(
                    f"{i}. {tool_call['tool_name']}({json.dumps(tool_call['parameters'])})"
                )

        if result.get("warning"):
            print(f"\nWarning: {result['warning']}")
        if result.get("error"):
            print(f"\nError: {result['error']}")

        if logger.isEnabledFor(logging.DEBUG) and result.get("history"):
            print("\n--- Full Conversation History (Debug) ---")
            print(json.dumps(result["history"], indent=2, default=str))

    except ValueError as ve:
        logger.error(f"Initialization error for StaticResearchAgent: {ve}")
    except Exception as e:
        logger.error(
            f"An error occurred during the agent example run: {e}", exc_info=True
        )
    finally:
        if agent:  # Check if agent was successfully initialized
            if hasattr(agent, "llm_client") and isinstance(
                agent.llm_client, AsyncOpenAI
            ):
                if hasattr(agent.llm_client, "aclose"):
                    await agent.llm_client.aclose()
                    logger.info("Agent's OpenAI client closed.")
            if (
                hasattr(agent, "retriever")
                and hasattr(agent.retriever, "es_client")
                and agent.retriever.es_client
            ):
                if hasattr(agent.retriever.es_client, "close"):
                    await agent.retriever.es_client.close()
                    logger.info("Agent's retriever's Elasticsearch client closed.")


if __name__ == "__main__":

    print("Running StaticResearchAgent example...")
    print("Please ensure your environment variables (e.g., OPENAI_API_KEY) are set.")
    asyncio.run(main_research_agent_example())
