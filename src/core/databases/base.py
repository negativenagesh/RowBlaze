from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class GraphDatabaseProvider(ABC):
    """Base class for graph database providers."""

    @abstractmethod
    async def connect(self, **kwargs) -> bool:
        """Connect to the database."""
        pass

    @abstractmethod
    async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a node in the graph database."""
        pass

    @abstractmethod
    async def create_relationship(
        self,
        from_node: str,
        to_node: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a relationship between two nodes."""
        pass

    @abstractmethod
    async def query(
        self, query_string: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query against the graph database."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        pass
