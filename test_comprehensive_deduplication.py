#!/usr/bin/env python3
"""
Comprehensive test script to verify that knowledge graph deduplication
handles various edge cases and variations in entity/relationship names.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.ingestion.rag_ingestion import ChunkingEmbeddingPDFProcessor


async def test_comprehensive_deduplication():
    """Test deduplication with various edge cases and name variations."""

    # Create a processor instance for testing
    params = {"model": "gpt-4o-mini", "index_name": "test"}
    config = {"api_key": "test"}
    processor = ChunkingEmbeddingPDFProcessor(params, config, None, ".pdf")

    print("=== Comprehensive Entity Deduplication Test ===")

    # Test entities with various name variations
    test_entities = [
        {"name": "John Smith", "type": "person", "description": "Software engineer"},
        {
            "name": "john smith",
            "type": "person",
            "description": "Developer",
        },  # Different case
        {
            "name": "John  Smith",
            "type": "person",
            "description": "Senior dev",
        },  # Extra spaces
        {
            "name": "John Smith.",
            "type": "person",
            "description": "Team lead",
        },  # With punctuation
        {
            "name": "J. Smith",
            "type": "person",
            "description": "Engineer",
        },  # Abbreviated
        {
            "name": "Microsoft Corp",
            "type": "organization",
            "description": "Tech company",
        },
        {
            "name": "Microsoft Corporation",
            "type": "organization",
            "description": "Software firm",
        },  # Different but similar
        {
            "name": "microsoft corp.",
            "type": "organization",
            "description": "Big tech",
        },  # Case + punctuation
        {
            "name": "AI/ML",
            "type": "technology",
            "description": "Artificial Intelligence",
        },
        {
            "name": "AI-ML",
            "type": "technology",
            "description": "Machine Learning",
        },  # Different separator
        {
            "name": "AIML",
            "type": "technology",
            "description": "AI and ML",
        },  # No separator
        {
            "name": "Python 3.9",
            "type": "technology",
            "description": "Programming language",
        },
        {
            "name": "Python v3.9",
            "type": "technology",
            "description": "Latest Python",
        },  # Version variation
    ]

    print(f"Original entities: {len(test_entities)}")
    for i, entity in enumerate(test_entities):
        print(
            f"  {i+1}. '{entity['name']}' ({entity['type']}) - {entity['description']}"
        )

    deduplicated_entities = processor._deduplicate_entities(test_entities)

    print(f"\nDeduplicated entities: {len(deduplicated_entities)}")
    for i, entity in enumerate(deduplicated_entities):
        print(
            f"  {i+1}. '{entity['name']}' ({entity['type']}) - {entity['description']}"
        )

    print("\n=== Comprehensive Relationship Deduplication Test ===")

    # Test relationships with various variations
    test_relationships = [
        {
            "source_entity": "John Smith",
            "target_entity": "Microsoft Corp",
            "relation": "works at",
            "relationship_description": "Employee",
            "relationship_weight": 0.8,
        },
        {
            "source_entity": "john smith",  # Different case
            "target_entity": "Microsoft Corp.",  # With punctuation
            "relation": "works-at",  # Different separator
            "relationship_description": "Employment",
            "relationship_weight": 0.9,
        },
        {
            "source_entity": "John Smith",
            "target_entity": "Python 3.9",
            "relation": "uses",
            "relationship_description": "Programming",
            "relationship_weight": 0.7,
        },
        {
            "source_entity": "Microsoft Corp",
            "target_entity": "AI/ML",
            "relation": "develops",
            "relationship_description": "Technology development",
            "relationship_weight": 0.6,
        },
        {
            "source_entity": "Microsoft Corporation",  # Variation
            "target_entity": "AI-ML",  # Variation
            "relation": "develops",
            "relationship_description": "AI research",
            "relationship_weight": 0.8,
        },
    ]

    print(f"Original relationships: {len(test_relationships)}")
    for i, rel in enumerate(test_relationships):
        print(
            f"  {i+1}. '{rel['source_entity']}' -> '{rel['relation']}' -> '{rel['target_entity']}' (weight: {rel.get('relationship_weight', 'N/A')})"
        )

    deduplicated_relationships = processor._deduplicate_relationships(
        test_relationships
    )

    print(f"\nDeduplicated relationships: {len(deduplicated_relationships)}")
    for i, rel in enumerate(deduplicated_relationships):
        print(
            f"  {i+1}. '{rel['source_entity']}' -> '{rel['relation']}' -> '{rel['target_entity']}' (weight: {rel.get('relationship_weight', 'N/A')})"
        )
        print(f"      Description: {rel.get('relationship_description', 'N/A')}")

    print("\n=== Comprehensive Hierarchy Deduplication Test ===")

    # Test hierarchies with variations
    test_hierarchies = [
        {
            "name": "Company Structure",
            "description": "Organizational hierarchy",
            "root_type": "organization",
            "levels": [
                {
                    "id": "1",
                    "name": "Executive Level",
                    "nodes": [{"id": "ceo", "name": "CEO"}],
                }
            ],
            "relationships": [],
        },
        {
            "name": "company structure",  # Different case
            "description": "Corporate org chart",
            "root_type": "Organization",  # Different case
            "levels": [
                {
                    "id": "2",
                    "name": "Management Level",
                    "nodes": [{"id": "manager", "name": "Manager"}],
                }
            ],
            "relationships": [],
        },
        {
            "name": "Company-Structure",  # With separator
            "description": "Business hierarchy",
            "root_type": "organization",
            "levels": [
                {
                    "id": "3",
                    "name": "Employee Level",
                    "nodes": [{"id": "employee", "name": "Employee"}],
                }
            ],
            "relationships": [],
        },
        {
            "name": "Technology Stack",
            "description": "Software architecture",
            "root_type": "technology",
            "levels": [
                {
                    "id": "1",
                    "name": "Backend",
                    "nodes": [{"id": "python", "name": "Python"}],
                }
            ],
            "relationships": [],
        },
    ]

    print(f"Original hierarchies: {len(test_hierarchies)}")
    for i, hierarchy in enumerate(test_hierarchies):
        print(
            f"  {i+1}. '{hierarchy['name']}' ({hierarchy['root_type']}) - {hierarchy['description']}"
        )
        print(f"      Levels: {len(hierarchy.get('levels', []))}")

    deduplicated_hierarchies = processor._deduplicate_hierarchies(test_hierarchies)

    print(f"\nDeduplicated hierarchies: {len(deduplicated_hierarchies)}")
    for i, hierarchy in enumerate(deduplicated_hierarchies):
        print(
            f"  {i+1}. '{hierarchy['name']}' ({hierarchy['root_type']}) - {hierarchy['description']}"
        )
        print(f"      Levels: {len(hierarchy.get('levels', []))}")

    print("\n=== Summary ===")

    entity_reduction = len(test_entities) - len(deduplicated_entities)
    relationship_reduction = len(test_relationships) - len(deduplicated_relationships)
    hierarchy_reduction = len(test_hierarchies) - len(deduplicated_hierarchies)

    print(f"Entity deduplication effectiveness:")
    print(
        f"  Original: {len(test_entities)} -> Deduplicated: {len(deduplicated_entities)} (reduced by {entity_reduction})"
    )

    print(f"Relationship deduplication effectiveness:")
    print(
        f"  Original: {len(test_relationships)} -> Deduplicated: {len(deduplicated_relationships)} (reduced by {relationship_reduction})"
    )

    print(f"Hierarchy deduplication effectiveness:")
    print(
        f"  Original: {len(test_hierarchies)} -> Deduplicated: {len(deduplicated_hierarchies)} (reduced by {hierarchy_reduction})"
    )

    print(
        f"\nTotal reduction: {entity_reduction + relationship_reduction + hierarchy_reduction} items"
    )

    # Test edge cases
    print("\n=== Edge Case Tests ===")

    # Empty lists
    empty_entities = processor._deduplicate_entities([])
    empty_relationships = processor._deduplicate_relationships([])
    empty_hierarchies = processor._deduplicate_hierarchies([])

    print(
        f"Empty list handling: entities={len(empty_entities)}, relationships={len(empty_relationships)}, hierarchies={len(empty_hierarchies)}"
    )

    # Entities with missing names
    entities_with_missing_names = [
        {"name": "", "type": "person", "description": "No name"},
        {"name": "Valid Name", "type": "person", "description": "Has name"},
        {"type": "person", "description": "Missing name key"},
    ]

    filtered_entities = processor._deduplicate_entities(entities_with_missing_names)
    print(
        f"Entities with missing names: {len(entities_with_missing_names)} -> {len(filtered_entities)} (filtered out {len(entities_with_missing_names) - len(filtered_entities)} invalid entries)"
    )

    print("\n=== Comprehensive Deduplication Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_comprehensive_deduplication())
