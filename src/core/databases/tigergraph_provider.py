from typing import Dict, List, Any, Optional, Union
import os
import httpx
from .base import GraphDatabaseProvider


class TigerGraphProvider(GraphDatabaseProvider):
    """TigerGraph database provider."""

    def __init__(self):
        self.host = None
        self.graph_name = None
        self.username = None
        self.password = None
        self.token = None
        self.api_base = None

    async def connect(self, **kwargs) -> bool:
        """Connect to TigerGraph."""
        self.host = kwargs.get("host") or os.getenv(
            "TIGERGRAPH_HOST", "http://localhost:9000"
        )
        self.graph_name = kwargs.get("graph") or os.getenv("TIGERGRAPH_GRAPH", "graph")
        self.username = kwargs.get("user") or os.getenv("TIGERGRAPH_USER", "tigergraph")
        self.password = kwargs.get("password") or os.getenv(
            "TIGERGRAPH_PASSWORD", "tigergraph"
        )

        self.api_base = f"{self.host}/restpp/v2/graph/{self.graph_name}"

        # Request a token
        try:
            auth_url = f"{self.host}/requesttoken"
            payload = {
                "graph": self.graph_name,
                "username": self.username,
                "password": self.password,
                "lifetime": 86400,  # 24 hours
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(auth_url, json=payload)
                response.raise_for_status()
                result = response.json()
                self.token = result.get("token")

                if not self.token:
                    return False

                return True
        except Exception as e:
            print(f"Error connecting to TigerGraph: {e}")
            self.token = None
            return False

    async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a node in TigerGraph."""
        if not self.token:
            raise RuntimeError("Not connected to TigerGraph")

        vertex_url = f"{self.api_base}/vertices/{label}"

        # Convert properties to string values as required by TigerGraph
        str_properties = {k: str(v) for k, v in properties.items()}

        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.token}"}
                response = await client.post(
                    vertex_url, json=str_properties, headers=headers
                )
                response.raise_for_status()
                result = response.json()

                # Return the generated vertex ID
                return result.get("results", [{}])[0].get("id", "unknown")
        except Exception as e:
            raise RuntimeError(f"Failed to create node in TigerGraph: {e}")

    async def create_relationship(
        self,
        from_node: str,
        to_node: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a relationship between nodes in TigerGraph."""
        if not self.token:
            raise RuntimeError("Not connected to TigerGraph")

        if not properties:
            properties = {}

        # TigerGraph uses edges in its API
        edge_url = f"{self.api_base}/edges/{relationship_type}"

        # Convert properties to string values
        str_properties = {k: str(v) for k, v in properties.items()}
        str_properties.update({"from_id": from_node, "to_id": to_node})

        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.token}"}
                response = await client.post(
                    edge_url, json=str_properties, headers=headers
                )
                response.raise_for_status()
                result = response.json()

                # TigerGraph doesn't return explicit edge IDs in the same way
                # This is a simplification, you'd typically use source+target+type
                return f"{from_node}>{relationship_type}>{to_node}"
        except Exception as e:
            raise RuntimeError(f"Failed to create relationship in TigerGraph: {e}")

    async def query(
        self, query_string: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a GSQL query against TigerGraph."""
        if not self.token:
            raise RuntimeError("Not connected to TigerGraph")

        if not params:
            params = {}

        # Using interpretation endpoint for GSQL
        query_url = f"{self.host}/gsqlserver/interpreted_query"

        payload = {"graph": self.graph_name, "query": query_string, "params": params}

        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.token}"}
                response = await client.post(query_url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json().get("results", [])
        except Exception as e:
            raise RuntimeError(f"Failed to execute query in TigerGraph: {e}")

    async def close(self) -> None:
        """Close the TigerGraph connection."""
        self.token = None
