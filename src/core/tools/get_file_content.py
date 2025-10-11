import logging
import json
from typing import Any, Dict, Optional, List

from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


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


class GetFileContentTool(Tool):
    """
    A tool to fetch and concatenate all text chunks for a given document_id
    from Elasticsearch.
    """

    def __init__(self):
        super().__init__(
            name="get_file_content",
            description=(
                "Fetches and concatenates all text chunks for a specified document_id from the knowledge base. "
                "Use this to retrieve the full available text content of a document when its ID is known."
            ),
            parameters={  # OpenAPI schema for parameters
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The unique ID of the document to fetch all content for.",
                    },
                },
                "required": ["document_id"],
            },
        )

    async def execute(
        self,
        document_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Fetches all text chunks for the given document_id from Elasticsearch
        and returns them as a single concatenated string.
        """
        logger.info(f"Executing {self.name} for document_id: {document_id}")

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

        if not document_id:
            logger.warning(f"document_id parameter is required for {self.name}.execute")
            return "Error: document_id parameter is required."

        max_chunks_to_fetch = 1000

        query_body = {
            "query": {"term": {"metadata.doc_id.keyword": document_id}},
            "sort": [
                {"metadata.page_number": "asc"},
                {"metadata.chunk_index_in_page": "asc"},
            ],
            "_source": ["chunk_text"],
            "size": max_chunks_to_fetch,
        }

        logger.debug(
            f"Elasticsearch query for {self.name}: {json.dumps(query_body, indent=2)}"
        )

        try:
            response = await es_client.search(index=index_name, body=query_body)
        except Exception as e:
            logger.error(
                f"Elasticsearch query failed for document_id '{document_id}': {e}",
                exc_info=True,
            )
            return (
                f"Error: Failed to query Elasticsearch for document_id '{document_id}'."
            )

        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            logger.warning(
                f"No content chunks found for document_id: {document_id} in index {index_name}."
            )
            return f"No content found for document ID: {document_id}."

        if len(hits) == max_chunks_to_fetch:
            logger.warning(
                f"Retrieved the maximum configured number of chunks ({max_chunks_to_fetch}) for document_id '{document_id}'. "
                "The document might be larger, and some content might be truncated."
            )

        all_chunk_texts: List[str] = [
            hit["_source"]["chunk_text"]
            for hit in hits
            if hit.get("_source", {}).get("chunk_text")
        ]

        if not all_chunk_texts:
            logger.warning(
                f"Found {len(hits)} hits for document_id '{document_id}', but no 'chunk_text' could be extracted."
            )
            return f"Content found for document ID: {document_id}, but text extraction failed."

        full_content = "\n\n---\n\n".join(all_chunk_texts)
        logger.info(
            f"Successfully retrieved and concatenated {len(all_chunk_texts)} chunks for document_id: {document_id}."
        )

        return full_content
