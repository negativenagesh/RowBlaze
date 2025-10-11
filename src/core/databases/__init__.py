from typing import Any, Dict, Optional

from .base import GraphDatabaseProvider
from .neo4j_provider import Neo4jProvider
from .tigergraph_provider import TigerGraphProvider


class DatabaseProviderFactory:
    """Factory for creating database providers."""

    providers = {"neo4j": Neo4jProvider, "tigergraph": TigerGraphProvider}

    @classmethod
    async def create_provider(
        cls, provider_type: str, **connection_params
    ) -> Optional[GraphDatabaseProvider]:
        """
        Create and initialize a database provider.

        Args:
            provider_type: Type of database provider ('neo4j' or 'tigergraph')
            **connection_params: Connection parameters specific to the provider

        Returns:
            Initialized provider or None if initialization failed
        """
        if provider_type not in cls.providers:
            raise ValueError(f"Unknown provider type: {provider_type}")

        provider_class = cls.providers[provider_type]
        provider = provider_class()

        success = await provider.connect(**connection_params)
        if not success:
            return None

        return provider


# Make it easy to import the factory
__all__ = ["DatabaseProviderFactory", "GraphDatabaseProvider"]
