import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Union
from pydantic import BaseModel, Field, validator
import os
from pathlib import Path


class OntologyVersion(BaseModel):
    """Version information for an ontology schema."""

    major: int = 0
    minor: int = 1
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "OntologyVersion":
        """Create version from string like '1.2.3'"""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))


class OntologyField(BaseModel):
    """Definition of a field in the ontology."""

    name: str
    type: str
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[str]] = None

    class Config:
        extra = "forbid"


class OntologyEntity(BaseModel):
    """Definition of an entity in the ontology."""

    name: str
    description: str
    fields: List[OntologyField]
    relationships: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "forbid"


class OntologySchema(BaseModel):
    """Complete ontology schema definition."""

    name: str
    version: OntologyVersion
    description: str
    entities: List[OntologyEntity]
    created: datetime = Field(default_factory=datetime.utcnow)
    updated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"

    def save_to_file(self, directory: str) -> str:
        """Save the ontology schema to a file."""
        os.makedirs(directory, exist_ok=True)
        version_str = str(self.version)
        filename = f"{self.name}-{version_str}.yaml"
        filepath = os.path.join(directory, filename)

        # Convert to dict and save as YAML
        with open(filepath, "w") as f:
            yaml.dump(self.dict(), f, sort_keys=False)

        return filepath

    @classmethod
    def load_from_file(cls, filepath: str) -> "OntologySchema":
        """Load ontology schema from file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)


class OntologyManager:
    """Manager for loading and versioning ontology schemas."""

    def __init__(self, schema_dir: str):
        self.schema_dir = schema_dir
        os.makedirs(schema_dir, exist_ok=True)
        self.schemas: Dict[str, Dict[str, OntologySchema]] = {}
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load all schemas from the schema directory."""
        schema_files = Path(self.schema_dir).glob("*.yaml")

        for schema_file in schema_files:
            try:
                schema = OntologySchema.load_from_file(str(schema_file))

                if schema.name not in self.schemas:
                    self.schemas[schema.name] = {}

                version_str = str(schema.version)
                self.schemas[schema.name][version_str] = schema
            except Exception as e:
                print(f"Error loading schema {schema_file}: {e}")

    def get_schema(
        self, name: str, version: Optional[str] = None
    ) -> Optional[OntologySchema]:
        """
        Get a schema by name and optionally version.
        If version is not specified, returns the latest version.
        """
        if name not in self.schemas:
            return None

        if version and version in self.schemas[name]:
            return self.schemas[name][version]

        # Return latest version
        versions = sorted(
            self.schemas[name].keys(), key=lambda v: OntologyVersion.from_string(v)
        )
        if not versions:
            return None

        return self.schemas[name][versions[-1]]

    def save_schema(self, schema: OntologySchema) -> str:
        """Save a schema to the schema directory."""
        filepath = schema.save_to_file(self.schema_dir)

        # Update in-memory cache
        version_str = str(schema.version)
        if schema.name not in self.schemas:
            self.schemas[schema.name] = {}

        self.schemas[schema.name][version_str] = schema

        return filepath

    def create_new_version(
        self, name: str, increment: str = "patch"
    ) -> Optional[OntologySchema]:
        """
        Create a new version of an existing schema.

        Args:
            name: Name of the schema to version
            increment: Type of version increment ('major', 'minor', or 'patch')

        Returns:
            New schema version or None if original schema not found
        """
        schema = self.get_schema(name)
        if not schema:
            return None

        # Create new version
        new_version = OntologyVersion(
            major=schema.version.major,
            minor=schema.version.minor,
            patch=schema.version.patch,
        )

        if increment == "major":
            new_version.major += 1
            new_version.minor = 0
            new_version.patch = 0
        elif increment == "minor":
            new_version.minor += 1
            new_version.patch = 0
        else:  # patch
            new_version.patch += 1

        # Create new schema with updated version
        new_schema = OntologySchema(
            name=schema.name,
            version=new_version,
            description=schema.description,
            entities=schema.entities,
            created=schema.created,
            updated=datetime.utcnow(),
        )

        # Save the new schema
        self.save_schema(new_schema)

        return new_schema
