#!/usr/bin/env python3
"""
Test script to verify knowledge graph deduplication functionality.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.ingestion.rag_ingestion import ChunkingEmbeddingPDFProcessor


def test_entity_deduplication():
    """Test entity deduplication logic."""
    print("=== Testing Entity Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Test data with duplicates
    entities = [
        {"name": "John Doe", "type": "Person", "description": "A software engineer"},
        {
            "name": "john doe",
            "type": "person",
            "description": "Works at TechCorp",
        },  # Duplicate with different case
        {
            "name": "TechCorp",
            "type": "Organization",
            "description": "A technology company",
        },
        {
            "name": "TechCorp",
            "type": "Organization",
            "description": "Founded in 2020",
        },  # Duplicate with additional info
        {
            "name": "Python",
            "type": "Programming Language",
            "description": "High-level programming language",
        },
        {"name": "Jane Smith", "type": "Person", "description": "Product manager"},
    ]

    print(f"Original entities: {len(entities)}")
    for i, entity in enumerate(entities):
        print(f"  {i+1}. {entity['name']} ({entity['type']}): {entity['description']}")

    # Apply deduplication
    deduplicated = processor._deduplicate_entities(entities)

    print(f"\nDeduplicated entities: {len(deduplicated)}")
    for i, entity in enumerate(deduplicated):
        print(f"  {i+1}. {entity['name']} ({entity['type']}): {entity['description']}")

    # Verify results
    assert (
        len(deduplicated) == 4
    ), f"Expected 4 unique entities, got {len(deduplicated)}"

    # Check that John Doe descriptions were merged
    john_doe = next((e for e in deduplicated if e["name"].lower() == "john doe"), None)
    assert john_doe is not None, "John Doe entity not found"
    assert (
        "software engineer" in john_doe["description"]
        and "TechCorp" in john_doe["description"]
    ), "John Doe descriptions not properly merged"

    # Check that TechCorp descriptions were merged
    techcorp = next((e for e in deduplicated if e["name"] == "TechCorp"), None)
    assert techcorp is not None, "TechCorp entity not found"
    assert (
        "technology company" in techcorp["description"]
        and "Founded in 2020" in techcorp["description"]
    ), "TechCorp descriptions not properly merged"

    print("✅ Entity deduplication test passed!")


def test_relationship_deduplication():
    """Test relationship deduplication logic."""
    print("\n=== Testing Relationship Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Test data with duplicates
    relationships = [
        {
            "source_entity": "John Doe",
            "target_entity": "TechCorp",
            "relation": "works_at",
            "relationship_description": "Employee relationship",
            "relationship_weight": 0.8,
        },
        {
            "source_entity": "john doe",  # Different case
            "target_entity": "techcorp",  # Different case
            "relation": "works_at",
            "relationship_description": "Full-time employee",
            "relationship_weight": 0.9,  # Higher weight
        },
        {
            "source_entity": "Jane Smith",
            "target_entity": "TechCorp",
            "relation": "works_at",
            "relationship_description": "Product manager role",
            "relationship_weight": 0.7,
        },
        {
            "source_entity": "TechCorp",
            "target_entity": "Python",
            "relation": "uses",
            "relationship_description": "Primary programming language",
            "relationship_weight": 0.6,
        },
    ]

    print(f"Original relationships: {len(relationships)}")
    for i, rel in enumerate(relationships):
        print(
            f"  {i+1}. {rel['source_entity']} --{rel['relation']}--> {rel['target_entity']} (weight: {rel['relationship_weight']})"
        )

    # Apply deduplication
    deduplicated = processor._deduplicate_relationships(relationships)

    print(f"\nDeduplicated relationships: {len(deduplicated)}")
    for i, rel in enumerate(deduplicated):
        print(
            f"  {i+1}. {rel['source_entity']} --{rel['relation']}--> {rel['target_entity']} (weight: {rel['relationship_weight']})"
        )
        print(f"      Description: {rel['relationship_description']}")

    # Verify results
    assert (
        len(deduplicated) == 3
    ), f"Expected 3 unique relationships, got {len(deduplicated)}"

    # Check that John Doe -> TechCorp relationship was merged with higher weight
    john_techcorp = next(
        (
            r
            for r in deduplicated
            if r["source_entity"].lower() == "john doe"
            and r["target_entity"].lower() == "techcorp"
        ),
        None,
    )
    assert john_techcorp is not None, "John Doe -> TechCorp relationship not found"
    assert john_techcorp["relationship_weight"] == 0.9, "Higher weight not preserved"
    assert (
        "Employee relationship" in john_techcorp["relationship_description"]
        and "Full-time employee" in john_techcorp["relationship_description"]
    ), "Relationship descriptions not properly merged"

    print("✅ Relationship deduplication test passed!")


def test_hierarchy_deduplication():
    """Test hierarchy deduplication logic."""
    print("\n=== Testing Hierarchy Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Test data with duplicates
    hierarchies = [
        {
            "name": "TechCorp Organization",
            "description": "Company structure",
            "root_type": "Organization",
            "levels": [
                {
                    "id": "1",
                    "name": "Company",
                    "nodes": [{"name": "TechCorp", "id": "tc1"}],
                },
                {
                    "id": "2",
                    "name": "Department",
                    "nodes": [{"name": "Engineering", "id": "eng1"}],
                },
            ],
            "relationships": [
                {
                    "type": "part_of",
                    "source": {"node_id": "eng1", "level": "2"},
                    "target": {"node_id": "tc1", "level": "1"},
                }
            ],
        },
        {
            "name": "techcorp organization",  # Different case
            "description": "Organizational hierarchy",
            "root_type": "organization",  # Different case
            "levels": [
                {
                    "id": "3",
                    "name": "Team",
                    "nodes": [{"name": "Backend Team", "id": "bt1"}],
                }  # Additional level
            ],
            "relationships": [
                {
                    "type": "reports_to",
                    "source": {"node_id": "bt1", "level": "3"},
                    "target": {"node_id": "eng1", "level": "2"},
                }
            ],
        },
        {
            "name": "Product Hierarchy",
            "description": "Product organization",
            "root_type": "Product",
            "levels": [
                {
                    "id": "1",
                    "name": "Product Line",
                    "nodes": [{"name": "Software Products", "id": "sp1"}],
                }
            ],
            "relationships": [],
        },
    ]

    print(f"Original hierarchies: {len(hierarchies)}")
    for i, hierarchy in enumerate(hierarchies):
        print(f"  {i+1}. {hierarchy['name']} ({hierarchy['root_type']})")
        print(
            f"      Levels: {len(hierarchy['levels'])}, Relationships: {len(hierarchy['relationships'])}"
        )

    # Apply deduplication
    deduplicated = processor._deduplicate_hierarchies(hierarchies)

    print(f"\nDeduplicated hierarchies: {len(deduplicated)}")
    for i, hierarchy in enumerate(deduplicated):
        print(f"  {i+1}. {hierarchy['name']} ({hierarchy['root_type']})")
        print(
            f"      Levels: {len(hierarchy['levels'])}, Relationships: {len(hierarchy['relationships'])}"
        )
        print(f"      Description: {hierarchy['description']}")

    # Verify results
    assert (
        len(deduplicated) == 2
    ), f"Expected 2 unique hierarchies, got {len(deduplicated)}"

    # Check that TechCorp hierarchies were merged
    techcorp_hierarchy = next(
        (h for h in deduplicated if h["name"].lower() == "techcorp organization"), None
    )
    assert techcorp_hierarchy is not None, "TechCorp hierarchy not found"
    assert len(techcorp_hierarchy["levels"]) == 3, "Levels not properly merged"
    assert (
        len(techcorp_hierarchy["relationships"]) == 2
    ), "Relationships not properly merged"
    assert (
        "Company structure" in techcorp_hierarchy["description"]
        and "Organizational hierarchy" in techcorp_hierarchy["description"]
    ), "Hierarchy descriptions not properly merged"

    print("✅ Hierarchy deduplication test passed!")


def test_empty_and_edge_cases():
    """Test edge cases for deduplication."""
    print("\n=== Testing Edge Cases ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Test empty lists
    assert processor._deduplicate_entities([]) == []
    assert processor._deduplicate_relationships([]) == []
    assert processor._deduplicate_hierarchies([]) == []

    # Test entities with missing names
    entities_with_missing = [
        {"name": "", "type": "Person", "description": "No name"},
        {"name": "Valid Entity", "type": "Organization", "description": "Has name"},
        {"type": "Person", "description": "Missing name key"},  # No name key
    ]

    deduplicated_entities = processor._deduplicate_entities(entities_with_missing)
    assert len(deduplicated_entities) == 1, "Should only keep entities with valid names"
    assert deduplicated_entities[0]["name"] == "Valid Entity"

    # Test relationships with missing fields
    relationships_with_missing = [
        {
            "source_entity": "",  # Empty source
            "target_entity": "Target",
            "relation": "relates_to",
        },
        {
            "source_entity": "Source",
            "target_entity": "Target",
            "relation": "relates_to",
        },
    ]

    deduplicated_relationships = processor._deduplicate_relationships(
        relationships_with_missing
    )
    assert (
        len(deduplicated_relationships) == 1
    ), "Should only keep complete relationships"

    print("✅ Edge cases test passed!")


def main():
    """Run all deduplication tests."""
    print("🧪 Running Knowledge Graph Deduplication Tests\n")

    try:
        test_entity_deduplication()
        test_relationship_deduplication()
        test_hierarchy_deduplication()
        test_empty_and_edge_cases()

        print("\n🎉 All deduplication tests passed successfully!")
        print("\nThe knowledge graph deduplication functionality is working correctly:")
        print("✅ Entities are deduplicated by name and type (case-insensitive)")
        print(
            "✅ Relationships are deduplicated by source, target, and relation (case-insensitive)"
        )
        print(
            "✅ Hierarchies are deduplicated by name and root_type (case-insensitive)"
        )
        print("✅ Descriptions are merged when duplicates are found")
        print("✅ Highest weights are preserved for relationships")
        print("✅ Edge cases are handled properly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
