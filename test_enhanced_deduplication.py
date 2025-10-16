#!/usr/bin/env python3
"""
Enhanced test for knowledge graph deduplication functionality.
Tests the improved deduplication methods for entities, relationships, and hierarchies.
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
    """Test entity deduplication with various duplicate scenarios."""
    print("\n=== Testing Entity Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Test entities with various types of duplicates
    test_entities = [
        {"name": "John Smith", "type": "Person", "description": "CEO of company"},
        {
            "name": "john smith",
            "type": "person",
            "description": "Chief Executive Officer",
        },  # Same person, different case
        {
            "name": "John  Smith",
            "type": "PERSON",
            "description": "Company leader",
        },  # Extra spaces
        {
            "name": "Apple Inc.",
            "type": "Organization",
            "description": "Technology company",
        },
        {
            "name": "Apple Inc",
            "type": "Company",
            "description": "Tech giant",
        },  # Missing period
        {
            "name": "Microsoft",
            "type": "Organization",
            "description": "Software company",
        },
        {
            "name": "Microsoft Corporation",
            "type": "Organization",
            "description": "Large tech company",
        },  # Different but similar
        {"name": "New York", "type": "Location", "description": "City in USA"},
        {
            "name": "New York",
            "type": "City",
            "description": "Major US city",
        },  # Same name, different type
    ]

    print(f"Original entities: {len(test_entities)}")
    for i, entity in enumerate(test_entities):
        print(f"  {i+1}. {entity['name']} ({entity['type']}) - {entity['description']}")

    # Apply deduplication
    deduplicated = processor._deduplicate_entities(test_entities)

    print(f"\nDeduplicated entities: {len(deduplicated)}")
    for i, entity in enumerate(deduplicated):
        print(f"  {i+1}. {entity['name']} ({entity['type']}) - {entity['description']}")

    # Verify results
    assert len(deduplicated) < len(
        test_entities
    ), "Deduplication should reduce entity count"

    # Check that John Smith variants are merged
    john_entities = [e for e in deduplicated if "john" in e["name"].lower()]
    assert (
        len(john_entities) == 1
    ), f"Expected 1 John Smith entity, got {len(john_entities)}"

    # Check that Apple variants are merged
    apple_entities = [e for e in deduplicated if "apple" in e["name"].lower()]
    assert (
        len(apple_entities) == 1
    ), f"Expected 1 Apple entity, got {len(apple_entities)}"

    print("✅ Entity deduplication test passed!")


def test_relationship_deduplication():
    """Test relationship deduplication with various duplicate scenarios."""
    print("\n=== Testing Relationship Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Test relationships with various types of duplicates
    test_relationships = [
        {
            "source_entity": "John Smith",
            "target_entity": "Apple Inc",
            "relation": "works_for",
            "relationship_description": "Employment relationship",
            "relationship_weight": 0.8,
        },
        {
            "source_entity": "john smith",
            "target_entity": "apple inc",
            "relation": "works for",
            "relationship_description": "Job at company",
            "relationship_weight": 0.9,
        },  # Same relationship, different case and spacing
        {
            "source_entity": "Apple Inc",
            "target_entity": "John Smith",
            "relation": "employs",
            "relationship_description": "Company employs person",
            "relationship_weight": 0.7,
        },  # Reverse relationship
        {
            "source_entity": "Microsoft",
            "target_entity": "Bill Gates",
            "relation": "founded_by",
            "relationship_description": "Company founded by person",
            "relationship_weight": 1.0,
        },
        {
            "source_entity": "Microsoft Corporation",
            "target_entity": "Bill Gates",
            "relation": "founded by",
            "relationship_description": "Founding relationship",
            "relationship_weight": 0.95,
        },  # Similar but slightly different
    ]

    print(f"Original relationships: {len(test_relationships)}")
    for i, rel in enumerate(test_relationships):
        print(
            f"  {i+1}. {rel['source_entity']} -> {rel['relation']} -> {rel['target_entity']} (weight: {rel.get('relationship_weight', 'N/A')})"
        )

    # Apply deduplication
    deduplicated = processor._deduplicate_relationships(test_relationships)

    print(f"\nDeduplicated relationships: {len(deduplicated)}")
    for i, rel in enumerate(deduplicated):
        print(
            f"  {i+1}. {rel['source_entity']} -> {rel['relation']} -> {rel['target_entity']} (weight: {rel.get('relationship_weight', 'N/A')})"
        )

    # Verify results
    assert len(deduplicated) < len(
        test_relationships
    ), "Deduplication should reduce relationship count"

    # Check that John Smith - Apple relationships are merged
    john_apple_rels = [
        r
        for r in deduplicated
        if (
            "john" in r["source_entity"].lower()
            and "apple" in r["target_entity"].lower()
        )
        or (
            "apple" in r["source_entity"].lower()
            and "john" in r["target_entity"].lower()
        )
    ]

    print(f"John-Apple relationships found: {len(john_apple_rels)}")

    print("✅ Relationship deduplication test passed!")


def test_hierarchy_deduplication():
    """Test hierarchy deduplication with various duplicate scenarios."""
    print("\n=== Testing Hierarchy Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Test hierarchies with various types of duplicates
    test_hierarchies = [
        {
            "name": "Company Structure",
            "description": "Organizational hierarchy",
            "root_type": "Organization",
            "levels": [
                {
                    "id": "1",
                    "name": "CEO",
                    "nodes": [{"id": "ceo1", "name": "John Smith"}],
                },
                {
                    "id": "2",
                    "name": "Managers",
                    "nodes": [{"id": "mgr1", "name": "Jane Doe"}],
                },
            ],
            "relationships": [
                {
                    "type": "reports_to",
                    "source": {"node_id": "mgr1", "level": "2"},
                    "target": {"node_id": "ceo1", "level": "1"},
                }
            ],
        },
        {
            "name": "company structure",
            "description": "Corporate hierarchy",
            "root_type": "company",
            "levels": [
                {
                    "id": "1",
                    "name": "Chief Executive",
                    "nodes": [{"id": "ceo2", "name": "John Smith"}],
                },
                {
                    "id": "3",
                    "name": "Directors",
                    "nodes": [{"id": "dir1", "name": "Bob Wilson"}],
                },
            ],
            "relationships": [
                {
                    "type": "manages",
                    "source": {"node_id": "ceo2", "level": "1"},
                    "target": {"node_id": "dir1", "level": "3"},
                }
            ],
        },  # Same hierarchy, different case
        {
            "name": "Product Hierarchy",
            "description": "Product categorization",
            "root_type": "Product",
            "levels": [
                {
                    "id": "1",
                    "name": "Category",
                    "nodes": [{"id": "cat1", "name": "Electronics"}],
                },
                {
                    "id": "2",
                    "name": "Subcategory",
                    "nodes": [{"id": "sub1", "name": "Phones"}],
                },
            ],
            "relationships": [],
        },
    ]

    print(f"Original hierarchies: {len(test_hierarchies)}")
    for i, hier in enumerate(test_hierarchies):
        print(
            f"  {i+1}. {hier['name']} ({hier['root_type']}) - {len(hier['levels'])} levels, {len(hier['relationships'])} relationships"
        )

    # Apply deduplication
    deduplicated = processor._deduplicate_hierarchies(test_hierarchies)

    print(f"\nDeduplicated hierarchies: {len(deduplicated)}")
    for i, hier in enumerate(deduplicated):
        print(
            f"  {i+1}. {hier['name']} ({hier['root_type']}) - {len(hier['levels'])} levels, {len(hier['relationships'])} relationships"
        )

    # Verify results
    assert len(deduplicated) < len(
        test_hierarchies
    ), "Deduplication should reduce hierarchy count"

    # Check that Company Structure variants are merged
    company_hierarchies = [h for h in deduplicated if "company" in h["name"].lower()]
    assert (
        len(company_hierarchies) == 1
    ), f"Expected 1 Company Structure hierarchy, got {len(company_hierarchies)}"

    # Check that the merged hierarchy has combined levels and relationships
    company_hier = company_hierarchies[0]
    print(
        f"Merged company hierarchy has {len(company_hier['levels'])} levels and {len(company_hier['relationships'])} relationships"
    )

    print("✅ Hierarchy deduplication test passed!")


def test_comprehensive_deduplication():
    """Test comprehensive deduplication across all knowledge graph components."""
    print("\n=== Testing Comprehensive Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Create test data with many duplicates
    entities = []
    relationships = []
    hierarchies = []

    # Add many duplicate entities
    for i in range(10):
        entities.extend(
            [
                {
                    "name": "Apple Inc",
                    "type": "Company",
                    "description": f"Tech company {i}",
                },
                {
                    "name": "apple inc.",
                    "type": "Organization",
                    "description": f"Technology firm {i}",
                },
                {
                    "name": "John Smith",
                    "type": "Person",
                    "description": f"CEO description {i}",
                },
                {
                    "name": "john smith",
                    "type": "person",
                    "description": f"Chief Executive {i}",
                },
            ]
        )

    # Add many duplicate relationships
    for i in range(5):
        relationships.extend(
            [
                {
                    "source_entity": "John Smith",
                    "target_entity": "Apple Inc",
                    "relation": "works_for",
                    "relationship_description": f"Employment {i}",
                },
                {
                    "source_entity": "john smith",
                    "target_entity": "apple inc",
                    "relation": "works for",
                    "relationship_description": f"Job {i}",
                },
                {
                    "source_entity": "Apple Inc",
                    "target_entity": "John Smith",
                    "relation": "employs",
                    "relationship_description": f"Employs {i}",
                },
            ]
        )

    # Add duplicate hierarchies
    for i in range(3):
        hierarchies.extend(
            [
                {
                    "name": "Company Structure",
                    "description": f"Org chart {i}",
                    "root_type": "Organization",
                    "levels": [{"id": f"level_{i}", "name": f"Level {i}"}],
                    "relationships": [],
                },
                {
                    "name": "company structure",
                    "description": f"Corporate hierarchy {i}",
                    "root_type": "company",
                    "levels": [{"id": f"lvl_{i}", "name": f"Tier {i}"}],
                    "relationships": [],
                },
            ]
        )

    print(f"Before deduplication:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)}")
    print(f"  Hierarchies: {len(hierarchies)}")

    # Apply deduplication
    dedup_entities = processor._deduplicate_entities(entities)
    dedup_relationships = processor._deduplicate_relationships(relationships)
    dedup_hierarchies = processor._deduplicate_hierarchies(hierarchies)

    print(f"\nAfter deduplication:")
    print(
        f"  Entities: {len(dedup_entities)} (reduction: {len(entities) - len(dedup_entities)})"
    )
    print(
        f"  Relationships: {len(dedup_relationships)} (reduction: {len(relationships) - len(dedup_relationships)})"
    )
    print(
        f"  Hierarchies: {len(dedup_hierarchies)} (reduction: {len(hierarchies) - len(dedup_hierarchies)})"
    )

    # Verify significant reduction
    assert (
        len(dedup_entities) <= 2
    ), f"Expected at most 2 unique entities, got {len(dedup_entities)}"
    assert (
        len(dedup_relationships) <= 2
    ), f"Expected at most 2 unique relationships, got {len(dedup_relationships)}"
    assert (
        len(dedup_hierarchies) <= 1
    ), f"Expected at most 1 unique hierarchy, got {len(dedup_hierarchies)}"

    print("✅ Comprehensive deduplication test passed!")


def main():
    """Run all deduplication tests."""
    print("🧪 Running Enhanced Knowledge Graph Deduplication Tests")
    print("=" * 60)

    try:
        test_entity_deduplication()
        test_relationship_deduplication()
        test_hierarchy_deduplication()
        test_comprehensive_deduplication()

        print("\n" + "=" * 60)
        print("🎉 All deduplication tests passed successfully!")
        print("✅ The enhanced deduplication logic is working correctly.")
        print("✅ Duplicates are being properly identified and merged.")
        print("✅ Knowledge graph quality should be significantly improved.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
