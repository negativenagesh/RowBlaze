import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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


class VectorSearchTool(Tool):
    """
    A tool to perform vector/semantic search over document chunks.
    Uses embeddings to find semantically similar content even when keywords don't match exactly.
    """

    def __init__(self):
        super().__init__(
            name="vector_search",
            description=(
                "Performs a semantic/vector search to find documents that are conceptually similar "
                "to your query, even if they don't use the same exact words. This tool uses "
                "embeddings to find content that matches the meaning of your question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for semantically similar content.",
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
        Executes a vector search using embeddings to find semantically similar document chunks.

        Args:
            query: The search query.
            top_k: The number of results to return.

        Returns:
            A formatted string with the search results.
        """
        logger.info(f"Executing {self.name} with query: '{query}' and top_k: {top_k}")

        if not self.context or not hasattr(self.context, "retriever"):
            msg = f"No context with a retriever provided for {self.name}. Cannot execute search."
            logger.error(msg)
            return f"Error: {msg}"

        retriever = self.context.retriever
        if not hasattr(retriever, "_generate_embedding") or not hasattr(
            retriever, "_semantic_search_chunks"
        ):
            msg = f"Retriever in context for {self.name} is missing required methods."
            logger.error(msg)
            return f"Error: {msg}"

        try:
            query_embedding_list = await retriever._generate_embedding([query])
            if not query_embedding_list or not query_embedding_list[0]:
                return f"Error: Could not generate embedding for the query '{query}'."

            query_embedding = query_embedding_list[0]

            results = await retriever._semantic_search_chunks(
                query_embedding, top_k=top_k
            )

            if not results:
                return f"No semantically similar content found for the query: '{query}'"

            formatted_lines = [
                f"Found {len(results)} semantically similar document chunks:"
            ]

            for i, res in enumerate(results):
                file_name = res.get("file_name", "Unknown")
                doc_id = res.get("doc_id", "Unknown")
                page = res.get("page_number", "N/A")
                score = res.get("score", 0.0)
                chunk_text = res.get("text", "").strip().replace("\n", " ")

                formatted_lines.append(
                    f"Document {i+1}: {file_name} (ID: {doc_id}, Page: {page}, Score: {score:.2f})"
                )
                formatted_lines.append(f'Content: "{chunk_text[:300]}..."\n')

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
            return f"Error executing vector search: {str(e)}"
