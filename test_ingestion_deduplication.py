#!/usr/bin/env python3
"""
Test to verify that deduplication is working in the actual ingestion pipeline.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.ingestion.rag_ingestion import ChunkingEmbeddingPDFProcessor


async def test_document_level_deduplication():
    """Test that document-level deduplication works correctly."""
    print("\n=== Testing Document-Level Deduplication ===")

    # Create a mock processor instance
    processor = ChunkingEmbeddingPDFProcessor(
        params={}, config={}, aclient_openai=None, file_extension=".pdf"
    )

    # Create mock processed chunks with duplicate KG data
    processed_chunks = []

    # Chunk 1 - has duplicate entities and relationships
    chunk1 = {
        "_source": {
            "chunk_text": "John Smith works for Apple Inc.",
            "metadata": {
                "file_name": "test.pdf",
                "entities": [
                    {"name": "John Smith", "type": "Person", "description": "CEO"},
                    {
                        "name": "Apple Inc",
                        "type": "Company",
                        "description": "Tech company",
                    },
                ],
                "relationships": [
                    {
                        "source_entity": "John Smith",
                        "target_entity": "Apple Inc",
                        "relation": "works_for",
                        "relationship_description": "Employment",
                    }
                ],
                "hierarchies": [
                    {
                        "name": "Company Structure",
                        "description": "Org chart",
                        "root_type": "Organization",
                        "levels": [],
                        "relationships": [],
                    }
                ],
            },
        }
    }

    # Chunk 2 - has same entities with slight variations
    chunk2 = {
        "_source": {
            "chunk_text": "john smith is employed by apple inc.",
            "metadata": {
                "file_name": "test.pdf",
                "entities": [
                    {
                        "name": "john smith",
                        "type": "person",
                        "description": "Chief Executive",
                    },
                    {
                        "name": "apple inc",
                        "type": "organization",
                        "description": "Technology firm",
                    },
                ],
                "relationships": [
                    {
                        "source_entity": "john smith",
                        "target_entity": "apple inc",
                        "relation": "works for",
                        "relationship_description": "Job",
                    }
                ],
                "hierarchies": [
                    {
                        "name": "company structure",
                        "description": "Corporate hierarchy",
                        "root_type": "company",
                        "levels": [],
                        "relationships": [],
                    }
                ],
            },
        }
    }

    # Chunk 3 - has more duplicates
    chunk3 = {
        "_source": {
            "chunk_text": "Apple Inc employs John Smith as CEO.",
            "metadata": {
                "file_name": "test.pdf",
                "entities": [
                    {
                        "name": "Apple Inc.",
                        "type": "Corporation",
                        "description": "Tech giant",
                    },
                    {
                        "name": "John  Smith",
                        "type": "Executive",
                        "description": "Company leader",
                    },
                ],
                "relationships": [
                    {
                        "source_entity": "Apple Inc",
                        "target_entity": "John Smith",
                        "relation": "employs",
                        "relationship_description": "Employment relationship",
                    }
                ],
                "hierarchies": [],
            },
        }
    }

    processed_chunks = [chunk1, chunk2, chunk3]

    print(f"Before deduplication:")
    total_entities = sum(
        len(chunk["_source"]["metadata"]["entities"]) for chunk in processed_chunks
    )
    total_relationships = sum(
        len(chunk["_source"]["metadata"]["relationships"]) for chunk in processed_chunks
    )
    total_hierarchies = sum(
        len(chunk["_source"]["metadata"]["hierarchies"]) for chunk in processed_chunks
    )

    print(f"  Total entities across all chunks: {total_entities}")
    print(f"  Total relationships across all chunks: {total_relationships}")
    print(f"  Total hierarchies across all chunks: {total_hierarchies}")

    # Apply document-level deduplication
    deduplicated_chunks = processor._apply_document_level_deduplication(
        processed_chunks, "test.pdf"
    )

    print(f"\nAfter deduplication:")
    dedup_entities = sum(
        len(chunk["_source"]["metadata"]["entities"]) for chunk in deduplicated_chunks
    )
    dedup_relationships = sum(
        len(chunk["_source"]["metadata"]["relationships"])
        for chunk in deduplicated_chunks
    )
    dedup_hierarchies = sum(
        len(chunk["_source"]["metadata"]["hierarchies"])
        for chunk in deduplicated_chunks
    )

    print(f"  Total entities across all chunks: {dedup_entities}")
    print(f"  Total relationships across all chunks: {dedup_relationships}")
    print(f"  Total hierarchies across all chunks: {dedup_hierarchies}")

    # Verify that deduplication occurred
    assert (
        dedup_entities < total_entities
    ), f"Expected entity reduction, got {dedup_entities} vs {total_entities}"
    assert (
        dedup_relationships < total_relationships
    ), f"Expected relationship reduction, got {dedup_relationships} vs {total_relationships}"
    assert (
        dedup_hierarchies < total_hierarchies
    ), f"Expected hierarchy reduction, got {dedup_hierarchies} vs {total_hierarchies}"

    # Check that entities are properly deduplicated and consistent
    chunk1_entities = deduplicated_chunks[0]["_source"]["metadata"]["entities"]
    chunk2_entities = deduplicated_chunks[1]["_source"]["metadata"]["entities"]
    chunk3_entities = deduplicated_chunks[2]["_source"]["metadata"]["entities"]

    print(f"\nEntity consistency check:")
    print(f"  Chunk 1 entities: {len(chunk1_entities)}")
    for entity in chunk1_entities:
        print(f"    - {entity['name']} ({entity['type']})")

    print(f"  Chunk 2 entities: {len(chunk2_entities)}")
    for entity in chunk2_entities:
        print(f"    - {entity['name']} ({entity['type']})")

    print(f"  Chunk 3 entities: {len(chunk3_entities)}")
    for entity in chunk3_entities:
        print(f"    - {entity['name']} ({entity['type']})")

    # Check that entities are properly deduplicated (same names should have consistent data)
    all_entities = chunk1_entities + chunk2_entities + chunk3_entities
    entity_names = [entity["name"] for entity in all_entities]
    unique_names = set(entity_names)

    print(f"\nDeduplication verification:")
    print(f"  Total entity instances: {len(all_entities)}")
    print(f"  Unique entity names: {len(unique_names)}")

    # Verify that each unique entity name appears with consistent data
    for name in unique_names:
        entities_with_name = [e for e in all_entities if e["name"] == name]
        if len(entities_with_name) > 1:
            # Check that all instances of this entity have the same type and description
            first_entity = entities_with_name[0]
            for entity in entities_with_name[1:]:
                assert (
                    entity["type"] == first_entity["type"]
                ), f"Entity '{name}' has inconsistent types: {entity['type']} vs {first_entity['type']}"
                # Description might be merged, so just check it's not empty
                assert entity[
                    "description"
                ], f"Entity '{name}' has empty description after deduplication"

    print("✅ Document-level deduplication test passed!")


def main():
    """Run the ingestion deduplication test."""
    print("🧪 Testing Deduplication in Ingestion Pipeline")
    print("=" * 50)

    try:
        asyncio.run(test_document_level_deduplication())

        print("\n" + "=" * 50)
        print("🎉 Ingestion deduplication test passed!")
        print("✅ Document-level deduplication is working correctly.")
        print("✅ Knowledge graph consistency is maintained across chunks.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
