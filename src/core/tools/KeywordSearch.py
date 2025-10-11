import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --- Base Tool Definition ---
class Tool:
    """Base class for tools that can be used by an agent."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.context: Optional[Any] = None

    def set_context(self, context: Any):
        self.context = context
        logger.debug(
            f"Context set for tool '{self.name}': {type(context).__name__ if context else 'None'}"
        )

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"The 'execute' method must be implemented by subclasses of Tool (e.g., {self.__class__.__name__})."
        )


# --- End Base Tool Definition ---


class KeywordSearchTool(Tool):
    """
    A tool to perform a keyword search for exact phrases within document contents.
    """

    def __init__(self):
        super().__init__(
            name="keyword_search",
            description=(
                "Performs a keyword search for an exact phrase within the content of stored documents. "
                "Use this when you need to find specific terms, names, or phrases as they are written."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact phrase or keyword to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The number of top results to return. Defaults to 5.",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(
        self, query: str, top_k: int = 5, *args: Any, **kwargs: Any
    ) -> str:
        """
        Performs a keyword search using the agent's retriever.

        Args:
            query: The keyword or phrase to search for.
            top_k: The number of results to return.

        Returns:
            A formatted string listing the relevant document chunks found.
        """
        logger.info(f"Executing {self.name} with query: '{query}' and top_k: {top_k}")

        if not self.context or not hasattr(self.context, "retriever"):
            msg = f"No context with a retriever provided for {self.name}. Cannot execute search."
            logger.error(msg)
            return f"Error: {msg}"

        retriever = self.context.retriever
        if not hasattr(retriever, "_keyword_search_chunks"):
            msg = f"Retriever in context for {self.name} is missing the '_keyword_search_chunks' method."
            logger.error(msg)
            return f"Error: {msg}"

        try:
            # 1. Perform keyword search using the retriever's method
            results: List[Dict[str, Any]] = await retriever._keyword_search_chunks(
                query, top_k=top_k
            )

            if not results:
                return f"No documents found containing the keyword/phrase: '{query}'"

            # 2. Format results for the LLM
            formatted_lines = [
                f"Found {len(results)} document chunks containing the keyword/phrase '{query}':"
            ]
            for res in results:
                file_name = res.get("file_name", "Unknown")
                doc_id = res.get("doc_id", "Unknown")
                page = res.get("page_number", "N/A")
                score = res.get("score", 0.0)
                chunk_text = res.get("text", "").strip().replace("\n", " ")
                formatted_lines.append(
                    f"- File: {file_name} (DocID: {doc_id}, Page: {page}, Score: {score:.3f})\n"
                    f'  Content: "{chunk_text[:200]}..."'
                )

            final_output = "\n".join(formatted_lines)
            logger.info(
                f"{self.name} executed successfully. Found {len(results)} results."
            )
            return final_output

        except Exception as e:
            logger.error(
                f"Error during {self.name} execution for query '{query}': {e}",
                exc_info=True,
            )
            return f"Error: Failed to execute keyword search: {str(e)}"
