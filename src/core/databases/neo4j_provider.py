from typing import Dict, List, Any, Optional, Union
import os
import neo4j
from .base import GraphDatabaseProvider

class Neo4jProvider(GraphDatabaseProvider):
    """Neo4j graph database provider."""
    
    def __init__(self):
        self.driver = None
        self.uri = None
        self.user = None
        self.password = None
    
    async def connect(self, **kwargs) -> bool:
        """Connect to Neo4j database."""
        self.uri = kwargs.get("uri") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = kwargs.get("user") or os.getenv("NEO4J_USER", "neo4j")
        self.password = kwargs.get("password") or os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = neo4j.AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection works
            await self.driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            self.driver = None
            return False
    
    async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a node in Neo4j."""
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")
        
        query = f"CREATE (n:{label} $props) RETURN id(n) as node_id"
        
        async with self.driver.session() as session:
            result = await session.run(query, props=properties)
            record = await result.single()
            return str(record["node_id"])
    
    async def create_relationship(self, from_node: str, to_node: str, 
                           relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """Create a relationship between two nodes."""
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")
        
        if not properties:
            properties = {}
        
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:{relationship_type} $props]->(b)
        RETURN id(r) as rel_id
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, from_id=int(from_node), 
                                      to_id=int(to_node), props=properties)
            record = await result.single()
            return str(record["rel_id"])
    
    async def query(self, query_string: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query against Neo4j."""
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")
        
        if not params:
            params = {}
        
        async with self.driver.session() as session:
            result = await session.run(query_string, **params)
            records = await result.values()
            return [dict(zip(result.keys(), record)) for record in records]
    
    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None