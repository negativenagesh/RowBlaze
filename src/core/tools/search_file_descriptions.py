import logging
import json
from typing import Any, Dict, Optional


# --- Base Tool Definition ---
# In a larger system, this Tool base class would ideally live in a shared abstractions module.
class Tool:
    """
    Base class for tools that can be used by an agent.
    """

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

from ...core.base.abstractions import AggregateSearchResult

logger = logging.getLogger(__name__)


class SearchFileDescriptionsTool(Tool):
    """
    A tool to search over high-level document data (AI-generated summaries).
    """

    def __init__(self):
        super().__init__(
            name="search_file_descriptions",
            description=(
                "Performs a full-text search over AI-generated summaries of stored documents. "
                "This does NOT retrieve chunk-level contents or knowledge-graph relationships. "
                "Use this when you need a broad overview of which documents (files) might be relevant to a topic."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string to search over available file summaries, e.g., 'list documents about financial regulations'.",
                    }
                },
                "required": ["query"],
            },
        )

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> str:
        """
        Performs a full-text search on the 'metadata.document_summary' field
        and returns a list of relevant documents.

        Args:
            query: The search query string.

        Returns:
            A formatted string listing the relevant documents found.
        """
        logger.info(f"Executing {self.name} with query: '{query}'")

        if not self.context or not hasattr(self.context, "retriever"):
            msg = f"No context with a retriever provided for {self.name}. Cannot execute search."
            logger.error(msg)
            return f"Error: {msg}"

        retriever = self.context.retriever
        if not hasattr(retriever, "es_client") or not hasattr(retriever, "index_name"):
            msg = f"Retriever in context for {self.name} is missing 'es_client' or 'index_name'."
            logger.error(msg)
            return f"Error: {msg}"

        es_client = retriever.es_client
        index_name = retriever.index_name

        # This query performs a full-text search on the summary and then aggregates
        # the results by document ID to get a list of relevant files.
        query_body = {
            "query": {
                "match": {
                    "metadata.document_summary": {"query": query, "operator": "and"}
                }
            },
            "aggs": {
                "relevant_documents": {
                    "terms": {
                        "field": "metadata.doc_id.keyword",
                        "size": 10,  # Return top 10 relevant documents
                    },
                    "aggs": {
                        "top_hit": {
                            "top_hits": {"size": 1, "_source": ["metadata.file_name"]}
                        },
                        "max_score": {"max": {"script": {"source": "_score"}}},
                    },
                }
            },
            "size": 0,
        }

        logger.debug(
            f"Executing file description search with query: {json.dumps(query_body, indent=2)}"
        )

        try:
            response = await es_client.search(index=index_name, body=query_body)

            buckets = (
                response.get("aggregations", {})
                .get("relevant_documents", {})
                .get("buckets", [])
            )

            if not buckets:
                return (
                    f"No documents found with summaries matching the query: '{query}'"
                )

            results = []
            for bucket in buckets:
                doc_id = bucket.get("key")
                score = bucket.get("max_score", {}).get("value", 0.0)
                top_hit = bucket.get("top_hit", {}).get("hits", {}).get("hits", [{}])[0]
                file_name = (
                    top_hit.get("_source", {})
                    .get("metadata", {})
                    .get("file_name", "Unknown")
                )
                results.append(
                    {"doc_id": doc_id, "file_name": file_name, "relevance_score": score}
                )

            # Sort by relevance score descending
            results.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Format for the LLM
            formatted_lines = ["Found relevant documents based on their summaries:"]
            for res in results:
                formatted_lines.append(
                    f"- Document ID: {res['doc_id']}, File Name: {res['file_name']} (Score: {res['relevance_score']:.2f})"
                )

            final_output = "\n".join(formatted_lines)
            logger.info(
                f"{self.name} executed successfully. Found {len(results)} relevant documents."
            )
            return final_output

        except Exception as e:
            logger.error(
                f"Error during {self.name} execution for query '{query}': {e}",
                exc_info=True,
            )
            return f"Error: Failed to execute file description search: {str(e)}"
