#!/usr/bin/env python3
"""
Test script to verify that knowledge graph deduplication is working properly
after the fixes to the ingestion pipeline.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.ingestion.rag_ingestion import ChunkingEmbeddingPDFProcessor


async def test_deduplication():
    """Test the deduplication methods with sample data."""

    # Create a processor instance for testing
    params = {"model": "gpt-4o-mini", "index_name": "test"}
    config = {"api_key": "test"}
    processor = ChunkingEmbeddingPDFProcessor(params, config, None, ".pdf")

    print("=== Testing Entity Deduplication ===")

    # Test entities with duplicates
    test_entities = [
        {"name": "John Smith", "type": "person", "description": "A software engineer"},
        {
            "name": "john smith",
            "type": "person",
            "description": "Works at Tech Corp",
        },  # Duplicate with different case
        {
            "name": "John  Smith",
            "type": "person",
            "description": "Senior developer",
        },  # Duplicate with extra spaces
        {
            "name": "Tech Corp",
            "type": "organization",
            "description": "A technology company",
        },
        {
            "name": "TechCorp",
            "type": "organization",
            "description": "Software development firm",
        },  # Similar but different
        {"name": "Python", "type": "technology", "description": "Programming language"},
        {
            "name": "Python",
            "type": "technology",
            "description": "Used for web development",
        },  # Exact duplicate
    ]

    print(f"Original entities: {len(test_entities)}")
    for i, entity in enumerate(test_entities):
        print(f"  {i+1}. {entity['name']} ({entity['type']}) - {entity['description']}")

    deduplicated_entities = processor._deduplicate_entities(test_entities)

    print(f"\nDeduplicated entities: {len(deduplicated_entities)}")
    for i, entity in enumerate(deduplicated_entities):
        print(f"  {i+1}. {entity['name']} ({entity['type']}) - {entity['description']}")

    print("\n=== Testing Relationship Deduplication ===")

    # Test relationships with duplicates
    test_relationships = [
        {
            "source_entity": "John Smith",
            "target_entity": "Tech Corp",
            "relation": "works at",
            "relationship_description": "Employee relationship",
            "relationship_weight": 0.8,
        },
        {
            "source_entity": "john smith",  # Different case
            "target_entity": "Tech Corp",
            "relation": "works at",
            "relationship_description": "Employment at the company",
            "relationship_weight": 0.9,
        },
        {
            "source_entity": "John Smith",
            "target_entity": "Python",
            "relation": "uses",
            "relationship_description": "Programming with Python",
            "relationship_weight": 0.7,
        },
        {
            "source_entity": "Tech Corp",
            "target_entity": "Python",
            "relation": "develops with",
            "relationship_description": "Company uses Python for development",
            "relationship_weight": 0.6,
        },
    ]

    print(f"Original relationships: {len(test_relationships)}")
    for i, rel in enumerate(test_relationships):
        print(
            f"  {i+1}. {rel['source_entity']} -> {rel['relation']} -> {rel['target_entity']} (weight: {rel.get('relationship_weight', 'N/A')})"
        )

    deduplicated_relationships = processor._deduplicate_relationships(
        test_relationships
    )

    print(f"\nDeduplicated relationships: {len(deduplicated_relationships)}")
    for i, rel in enumerate(deduplicated_relationships):
        print(
            f"  {i+1}. {rel['source_entity']} -> {rel['relation']} -> {rel['target_entity']} (weight: {rel.get('relationship_weight', 'N/A')})"
        )
        print(f"      Description: {rel.get('relationship_description', 'N/A')}")

    print("\n=== Testing Hierarchy Deduplication ===")

    # Test hierarchies with duplicates
    test_hierarchies = [
        {
            "name": "Company Structure",
            "description": "Organizational hierarchy",
            "root_type": "organization",
            "levels": [
                {
                    "id": "1",
                    "name": "Executive",
                    "nodes": [{"id": "ceo", "name": "CEO"}],
                }
            ],
            "relationships": [],
        },
        {
            "name": "company structure",  # Different case
            "description": "Corporate hierarchy",
            "root_type": "organization",
            "levels": [
                {
                    "id": "2",
                    "name": "Management",
                    "nodes": [{"id": "manager", "name": "Manager"}],
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
            f"  {i+1}. {hierarchy['name']} ({hierarchy['root_type']}) - {hierarchy['description']}"
        )
        print(f"      Levels: {len(hierarchy.get('levels', []))}")

    deduplicated_hierarchies = processor._deduplicate_hierarchies(test_hierarchies)

    print(f"\nDeduplicated hierarchies: {len(deduplicated_hierarchies)}")
    for i, hierarchy in enumerate(deduplicated_hierarchies):
        print(
            f"  {i+1}. {hierarchy['name']} ({hierarchy['root_type']}) - {hierarchy['description']}"
        )
        print(f"      Levels: {len(hierarchy.get('levels', []))}")

    print("\n=== Testing Document-Level Deduplication ===")

    # Test document-level deduplication with sample chunks
    test_chunks = [
        {
            "_source": {
                "chunk_text": "Chunk 1 content",
                "metadata": {
                    "entities": [
                        {
                            "name": "John Smith",
                            "type": "person",
                            "description": "Engineer",
                        },
                        {
                            "name": "Tech Corp",
                            "type": "organization",
                            "description": "Company",
                        },
                    ],
                    "relationships": [
                        {
                            "source_entity": "John Smith",
                            "target_entity": "Tech Corp",
                            "relation": "works at",
                            "relationship_description": "Employment",
                        }
                    ],
                    "hierarchies": [
                        {
                            "name": "Company Structure",
                            "root_type": "organization",
                            "description": "Org chart",
                            "levels": [],
                            "relationships": [],
                        }
                    ],
                },
            }
        },
        {
            "_source": {
                "chunk_text": "Chunk 2 content",
                "metadata": {
                    "entities": [
                        {
                            "name": "john smith",
                            "type": "person",
                            "description": "Developer",
                        },  # Duplicate
                        {
                            "name": "Python",
                            "type": "technology",
                            "description": "Language",
                        },
                    ],
                    "relationships": [
                        {
                            "source_entity": "john smith",  # Duplicate
                            "target_entity": "Tech Corp",
                            "relation": "works at",
                            "relationship_description": "Employee",
                        }
                    ],
                    "hierarchies": [
                        {
                            "name": "company structure",  # Duplicate
                            "root_type": "organization",
                            "description": "Corporate hierarchy",
                            "levels": [],
                            "relationships": [],
                        }
                    ],
                },
            }
        },
    ]

    print(f"Original chunks: {len(test_chunks)}")
    total_entities = sum(
        len(chunk["_source"]["metadata"]["entities"]) for chunk in test_chunks
    )
    total_relationships = sum(
        len(chunk["_source"]["metadata"]["relationships"]) for chunk in test_chunks
    )
    total_hierarchies = sum(
        len(chunk["_source"]["metadata"]["hierarchies"]) for chunk in test_chunks
    )

    print(f"  Total entities across chunks: {total_entities}")
    print(f"  Total relationships across chunks: {total_relationships}")
    print(f"  Total hierarchies across chunks: {total_hierarchies}")

    deduplicated_chunks = processor._apply_document_level_deduplication(
        test_chunks, "test_file.pdf"
    )

    print(f"\nAfter document-level deduplication:")
    total_entities_after = sum(
        len(chunk["_source"]["metadata"]["entities"]) for chunk in deduplicated_chunks
    )
    total_relationships_after = sum(
        len(chunk["_source"]["metadata"]["relationships"])
        for chunk in deduplicated_chunks
    )
    total_hierarchies_after = sum(
        len(chunk["_source"]["metadata"]["hierarchies"])
        for chunk in deduplicated_chunks
    )

    print(f"  Total entities across chunks: {total_entities_after}")
    print(f"  Total relationships across chunks: {total_relationships_after}")
    print(f"  Total hierarchies across chunks: {total_hierarchies_after}")

    # Show detailed results
    print("\nDetailed chunk contents after deduplication:")
    for i, chunk in enumerate(deduplicated_chunks):
        metadata = chunk["_source"]["metadata"]
        print(f"\nChunk {i+1}:")
        print(f"  Entities ({len(metadata['entities'])}):")
        for entity in metadata["entities"]:
            print(f"    - {entity['name']} ({entity['type']})")
        print(f"  Relationships ({len(metadata['relationships'])}):")
        for rel in metadata["relationships"]:
            print(
                f"    - {rel['source_entity']} -> {rel['relation']} -> {rel['target_entity']}"
            )
        print(f"  Hierarchies ({len(metadata['hierarchies'])}):")
        for hierarchy in metadata["hierarchies"]:
            print(f"    - {hierarchy['name']} ({hierarchy['root_type']})")

    print("\n=== Deduplication Test Complete ===")

    # Summary
    entity_reduction = total_entities - total_entities_after
    relationship_reduction = total_relationships - total_relationships_after
    hierarchy_reduction = total_hierarchies - total_hierarchies_after

    print(f"\nSummary of deduplication effectiveness:")
    print(
        f"  Entities reduced by: {entity_reduction} ({entity_reduction/total_entities*100:.1f}%)"
    )
    print(
        f"  Relationships reduced by: {relationship_reduction} ({relationship_reduction/total_relationships*100:.1f}%)"
    )
    print(
        f"  Hierarchies reduced by: {hierarchy_reduction} ({hierarchy_reduction/total_hierarchies*100:.1f}%)"
    )


if __name__ == "__main__":
    asyncio.run(test_deduplication())
