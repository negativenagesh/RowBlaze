import os
import asyncio
import yaml
import pymongo
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback
import requests
import time
import httpx
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

LOCAL_RERANKER_PATH = "./bge-reranker-base" 

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import TransportError
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import nltk

from sdk.response import FunctionResponse, Messages
from sdk.message import Message

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_INSTALLED = True
except ImportError:
    SENTENTCE_TRANSFORMERS_INSTALLED = False

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = 3072
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-base")
ELASTICSEARCH_URL = os.getenv("RAG_UPLOAD_ELASTIC_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
ES_USER = os.getenv("ELASTIC_USER")                    
ES_PASS = os.getenv("ELASTIC_PASSWORD")

PROVENCE_MODEL_ID = "naver/provence-reranker-debertav3-v1"
LOCAL_PROVENCE_PATH = "./provence-model"
LOCAL_NLTK_PATH = "./nltk_data"

async def init_async_openai_client() -> Optional[AsyncOpenAI]:
  openai_api_key = os.getenv("OPEN_AI_KEY")
  if not openai_api_key:
    print("❌ OPENAI_API_KEY not found in .env. OpenAI client will not be functional.")
    return None

  try:
    client = AsyncOpenAI(api_key=openai_api_key)
    print("✅ AsyncOpenAI client initialized.")
    return client
  except Exception as e:
    print(f"❌ Failed to initialize AsyncOpenAI client: {e}")
    return None

async def check_async_elasticsearch_connection() -> Optional[AsyncElasticsearch]:
  try:
    es_client = None
    if os.getenv("ELASTICSEARCH_API_KEY"):
        es_client = AsyncElasticsearch(
                ELASTICSEARCH_URL,
                api_key=ELASTICSEARCH_API_KEY,
                request_timeout=60,
                retry_on_timeout=True
            )

    if not await es_client.ping():
        print("❌ Ping to Elasticsearch cluster failed. URL may be incorrect or server is down.")
        return None

    print("✅ AsyncElasticsearch client initialized.")
    return es_client
  except Exception as e:
    print(f"❌ Failed to initialize AsyncElasticsearch client: {e}")
    return None

SYSTEM_PROMPT_TEMPLATE = """
**SYSTEM PROMPT**

## YOUR ROLE
You are a sophisticated AI assistant with expertise in analyzing and synthesizing information from provided documents. Your primary function is to answer user questions accurately and comprehensively, based *exclusively* on the context provided to you.

## TASK
Your task is to generate a detailed, well-structured, and definitive answer to the user's original query. You must use the information presented in the `CONTEXT` section below. The context contains retrieved text chunks and structured data from a knowledge graph, all extracted from relevant documents.

## INSTRUCTIONS
1.  **Synthesize, Don't Just List**: Do not simply list the retrieved information. Synthesize the data from the various chunks and knowledge graph entries into a coherent, flowing answer.
2.  **Strictly Context-Bound**: Your answer **MUST** be based solely on the provided `CONTEXT`. Do not use any external knowledge or make assumptions beyond what is written in the text.
3.  **Acknowledge Insufficiency**: If the provided context does not contain enough information to answer the question fully, explicitly state that the answer cannot be found in the provided documents. Do not attempt to guess.
4.  **Structured Formatting**: Present your answer in a clear and organized manner. Use markdown for formatting:
    *   Use headings (`##`, `###`) to structure your response.
    *   Use bullet points (`*` or `-`) for lists.
    *   Use bolding (`**text**`) for key terms and concepts.
    *   Use tables to present structured data or comparisons where appropriate.
5.  **Cite Your Sources**: At the end of **each paragraph** in your answer, you **MUST** add a citation referencing the source file. Use the `File` name provided in the context for the information used in that paragraph. For example: `This is a paragraph summarizing information. [some_document.pdf]`. If a paragraph synthesizes information from multiple files, cite all of them, like `[file_one.pdf, file_two.docx]`. This is crucial for traceability.

---

**CONTEXT BEGINS**

{context}

**CONTEXT ENDS**

---

Based on the context above, provide a comprehensive and well-structured final answer to the original user query. Remember to cite your sources as instructed.

**Final Answer:**
# """

USER_PROMPT_TEMPLATE = """
Following all the rules, constraints, and the step-by-step methodology defined in your system role, provide a direct and clear answer to my original question based *only* on the context below.

**Original Question:** "{original_query}"

---
**CONTEXT BEGINS**

{context}

**CONTEXT ENDS**
---

**Final Answer:**
"""
class RAGFusionRetriever:
    def __init__(self, params: Any, config: Any, es_client: Any, aclient_openai: Optional[AsyncOpenAI]):
        self.aclient_openai = aclient_openai
        self.params = params
        self.config = config
        self.es_client = es_client
        self.rag_fusion_prompt_template = self._load_prompt_template("rag_fusion")

        self.reranker = None
        self.provence_pruner = None
        self.deep_research = self.params.get('deep_research', False)
        self.embedding_model = None
        self.embedding_dims = OPENAI_EMBEDDING_DIMENSIONS
        self.reranker = None
        
        if self.deep_research:
            try:
                print(f"Deep research enabled. Initializing reranker from local path: {LOCAL_RERANKER_PATH}")
                self.reranker = CrossEncoder(LOCAL_RERANKER_PATH)
                print(f"CrossEncoder reranker initialized successfully from: {LOCAL_RERANKER_PATH}")
            except Exception as e:
                print(f"Failed to initialize CrossEncoder reranker from local path {LOCAL_RERANKER_PATH}: {e}")

    def _load_prompt_template(self, prompt_name: str) -> str:
        try:
            prompt_file_path = PROMPTS_DIR / f"{prompt_name}.yaml"
            with open(prompt_file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)

            if prompt_data and prompt_name in prompt_data and "template" in prompt_data[prompt_name]:
                template_content = prompt_data[prompt_name]["template"]
                print(f"Successfully loaded prompt template for '{prompt_name}'.")
                return template_content
            else:
                print(f"Prompt template for '{prompt_name}' not found or invalid in {prompt_file_path}.")
                raise ValueError(f"Invalid prompt structure for {prompt_name}")
        except FileNotFoundError:
            print(f"Prompt file not found: {prompt_file_path}")
            raise
        except Exception as e:
            print(f"Error loading prompt '{prompt_name}': {e}")
            raise
    
    async def _fetch_schema_chunks_by_file(self) -> Dict[str, str]:
        """
        Returns a mapping: file_name -> first chunk_text for that file (for .csv/.xlsx files only).
        """
        index_name = self.config.get('index_name')
        schema_chunks = {}
        try:
            response = await self.es_client.search(
                index=index_name,
                size=1000,
                query={
                    "bool": {
                        "should": [
                            {"wildcard": {"metadata.file_name.keyword": "*.csv"}},
                            {"wildcard": {"metadata.file_name.keyword": "*.xlsx"}},
                            {"wildcard": {"metadata.file_name.keyword": "*.pdf"}},
                            {"wildcard": {"metadata.file_name.keyword": "*.txt"}},
                        ]
                    }
                },
                _source_includes=["chunk_text", "metadata.file_name"]
            )
            seen_files = set()
            for hit in response.get('hits', {}).get('hits', []):
                file_name = hit.get('_source', {}).get('metadata', {}).get('file_name')
                if file_name and file_name not in seen_files:
                    schema_chunks[file_name] = hit.get('_source', {}).get('chunk_text', '')
                    seen_files.add(file_name)
            print(f"Schema chunks for most relevent keyword extraction: {schema_chunks}")
            return schema_chunks
        except Exception as e:
            print(f"❌ Error fetching schema chunks by file: {e}")
            return {}
    
    async def _call_openai_api(
        self,
        model_name: str,
        payload_messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """A unified async method to call OpenAI text models with retry logic."""
        if not self.aclient_openai:
            print("❌ OpenAI client not configured. Cannot make API call.")
            return ""

        print(f"*** openai messages: {payload_messages}")
        max_retries = 3
        base_delay_seconds = 10

        for attempt in range(max_retries):
            try:
                start_time = datetime.now(timezone.utc)
                response = await self.aclient_openai.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    max_tokens=max_tokens,
                )
                
                content = response.choices[0].message.content

                if content:
                    print(f"OpenAI API call successful. Preview: {content}")
                    return content
                else:
                    print(f"OpenAI API returned empty content. Attempt {attempt + 1}/{max_retries}")

            except Exception as e:
                print(f"OpenAI API call failed (Attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt + 1 < max_retries:
                delay = base_delay_seconds * (2 ** attempt)
                print(f"Waiting for {delay} seconds before retrying...")
                await asyncio.sleep(delay)

        print("Max retries reached for OpenAI API call. Returning empty string.")
        return ""
    
    async def _classify_query_type(self, query: str) -> str:
        """
        Uses an LLM to classify the user's query into a specific category.
        """
        classification_prompt = f"""
        You are an expert query classifier. Your task is to classify the user's query into one of the following categories based on its intent. Respond with ONLY the category name.

        Categories:
        - `factual_lookup`: For queries asking for specific, discrete pieces of information, like a number, name, date, or a specific definition. Example: "What is the lease start date for Tenant X?" or "How many floors does the building have?"
        - `summary_extraction`: For queries that ask for a summary of a topic, document, or concept. Example: "Summarize the main points of the contract." or "Provide an overview of the quarterly financial report."
        - `comparison`: For queries that ask to compare two or more items. Example: "Compare the lease terms for Tenant A and Tenant B."
        - `complex_analysis`: For broad, open-ended questions that require synthesizing information from multiple sources or analyzing relationships. This is the default for queries that don't fit other categories. Example: "What are the potential risks associated with the new project?"

        User Query: "{query}"

        Category:
        """
        valid_types = ['factual_lookup', 'summary_extraction', 'comparison', 'complex_analysis']
        
        query_type = ""
        print(f"Classifying query via OpenAI: '{query}'")
        messages = [{"role": "user", "content": classification_prompt}]
        query_type = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            max_tokens=20,
            temperature=0.0
        )

        cleaned_query_type = query_type.strip().lower()
        if cleaned_query_type in valid_types:
            print(f"✅ Query classified as: '{cleaned_query_type}'")
            return cleaned_query_type
        else:
            print(f"⚠️ Warning: LLM returned invalid query type '{cleaned_query_type}'. Defaulting to 'complex_analysis'.")
            return 'complex_analysis'

    async def _prune_documents(self, query: str, documents: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        if not self.provence_pruner:
            print("Provence pruner not initialized. Skipping pruning.")
            return documents
        if not documents:
            print(f"No {doc_type} documents to prune for query: '{query[:50]}...'")
            return []

        print(f"Pruning {len(documents)} {doc_type} documents with Provence for query: '{query[:50]}...'")
        
        text_key = "text" if doc_type == "chunk" else "chunk_text"
        
        pruned_docs = []
        for doc_idx,doc in enumerate(documents):
            original_text = doc.get(text_key, "")
            if not original_text:
                pruned_docs.append(doc) # Keep doc as is if no text
                continue

            try:
                # Run synchronous model inference in a thread
                provence_output = await asyncio.to_thread(
                    self.provence_pruner.process,
                    question=query,
                    context=original_text,
                    threshold=0.1, # Recommended conservative threshold
                    always_select_title=True
                )
                pruned_text = provence_output.get('pruned_context', original_text)
                
                doc_copy = doc.copy()
                doc_copy[text_key] = pruned_text
                # Optionally store the provence score if needed later
                doc_copy['provence_score'] = provence_output.get('reranking_score')
                pruned_docs.append(doc_copy)
                
                if len(original_text) != len(pruned_text):
                    print(f"  - Pruned content for doc #{doc_idx+1} (File: {doc.get('file_name', 'N/A')}, Page: {doc.get('page_number', 'N/A')}). Original len: {len(original_text)}, Pruned len: {len(pruned_text)}")
                else:
                    print(f"  - No content pruned for doc #{doc_idx+1} (File: {doc.get('file_name', 'N/A')}, Page: {doc.get('page_number', 'N/A')}).")
                
                print(f"    - Pruned Text: '{pruned_text}'")
                
            except Exception as e:
                print(f"  - Error pruning document with Provence: {e}. Using original text.")
                pruned_docs.append(doc) # Append original doc on error

        print(f"Successfully pruned {len(documents)} {doc_type} documents.")
        return pruned_docs
    
    async def _generate_subqueries(self, original_query: str,num_subqueries: int = 2) -> List[str]:
        
        if not self.rag_fusion_prompt_template:
            print("RAG Fusion prompt template not loaded. Cannot generate subqueries.")
            return []

        formatted_prompt = self.rag_fusion_prompt_template.format(
            num_outputs=num_subqueries, message=original_query
        )
        messages = [
            {"role": "system", "content": "You are an AI assistant that generates multiple search queries based on a single user query."},
            {"role": "user", "content": formatted_prompt}
        ]
        
        llm_response_content = ""
        
        print(f"Generating {num_subqueries} subqueries via OpenAI for: '{original_query}'")
        llm_response_content = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            max_tokens=500,
            temperature=0.5
        )

        if not llm_response_content:
            print("LLM returned empty content for subqueries.")
            return []

        splitter = '\n\n'
        subqueries = [sq.strip() for sq in llm_response_content.split(splitter) if sq.strip()]
        print(f"✅ Successfully generated {len(subqueries)} subqueries: {subqueries}")
        return subqueries[:num_subqueries]
    
    async def _generate_embedding(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        
        if not self.aclient_openai:
            print("OpenAI client not available. Cannot generate embeddings.")
            return [[] for _ in texts] 
        
        all_embeddings = []
        openai_batch_size = 2048 
        try:
            for i in range(0, len(texts), openai_batch_size):
                batch_texts = texts[i:i + openai_batch_size]
                processed_batch_texts = [text if text.strip() else " " for text in batch_texts]
                
                response = await self.aclient_openai.embeddings.create(
                    input=processed_batch_texts, 
                    model=OPENAI_EMBEDDING_MODEL, 
                    dimensions=self.embedding_dims
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [[] for _ in texts]

    async def _semantic_search_chunks(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        if not query_embedding:
            print("Semantic search skipped: No query embedding provided.")
            return []

        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10
        }
        index_name = self.config.get('index_name')
        print('index name in _semantic_search_chunks;-', index_name)
        # print(f"Performing semantic search for chunks with top_k={top_k}. Query: {json.dumps(knn_query)}")
        try:
            response = await self.es_client.search(
                index=index_name,
                knn=knn_query,
                size=top_k,
                _source_includes=["chunk_text", "metadata.file_name", "metadata.doc_id", "metadata.page_number", "metadata.chunk_index_in_page"]
            )
            results = []
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata = source.get('metadata', {})
                results.append({
                    "id":hit.get('_id'),
                    "text": source.get('chunk_text'),
                    "score": hit.get('_score'),
                    "file_name": metadata.get('file_name'),
                    "doc_id": metadata.get('doc_id'),
                    "page_number": metadata.get('page_number'),
                    "chunk_index_in_page": metadata.get('chunk_index_in_page')
                })
            print(f"Semantic search found {len(results)} chunks.")
            return results
        except TransportError as e:
            print(f"Elasticsearch semantic search error: {e}")
            return []

    async def _structured_kg_search(self, subquery_embedding: List[float], top_k: int = 3,top_k_entities: int = 5) -> List[Dict[str, Any]]:
        
        """
        Performs a semantic search on nested entity description embeddings, then filters
        for the top N entities within each returned document based on cosine similarity.
        """
        if not subquery_embedding:
            print("Structured KG search skipped: No subquery embedding provided.")
            return []
        
        knn_query = {
            "field": "metadata.entities.description_embedding",
            "query_vector": subquery_embedding,
            "k": top_k,
            "num_candidates": top_k * 10,
            "filter": {
                "bool": {
                    "must_not": [
                        {"wildcard": {"metadata.file_name.keyword": "*.xlsx"}},
                        {"wildcard": {"metadata.file_name.keyword": "*.csv"}}
                    ]
                }
            }
        }
        index_name = self.config.get('index_name')
        print('index name in _structured_kg_search:-', index_name)
        
        # print(f"Performing structured KG search, KNN Query: {json.dumps(knn_query,indent=2)}")
        try:
            response = await self.es_client.search(
                index=index_name,
                knn=knn_query,
                size=top_k,
                _source=["chunk_text", "metadata"]
            )
            
            final_results=[]
            query_vec=np.array(subquery_embedding)
            
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata_from_hit = source.get('metadata', {})
                all_entities = metadata_from_hit.get('entities', [])
                all_relationships = metadata_from_hit.get('relationships', [])

                if not all_entities:
                    continue

                # Score each entity in the document against the subquery embedding
                scored_entities = []
                for entity in all_entities:
                    entity_embedding_list = entity.get("description_embedding")
                    if entity_embedding_list:
                        entity_vec = np.array(entity_embedding_list)
                        # Calculate cosine similarity, handle potential norm=0
                        norm_query = np.linalg.norm(query_vec)
                        norm_entity = np.linalg.norm(entity_vec)
                        if norm_query > 0 and norm_entity > 0:
                            similarity = np.dot(query_vec, entity_vec) / (norm_query * norm_entity)
                            scored_entities.append((similarity, entity))

                # Sort entities by score and take top N
                scored_entities.sort(key=lambda x: x[0], reverse=True)
                top_entities = [entity for score, entity in scored_entities[:top_k_entities]]
                top_entity_names = {entity['name'] for entity in top_entities if entity.get('name')}

                # Filter relationships based on the top entities
                filtered_relationships = []
                if top_entity_names:
                    for rel in all_relationships:
                        if rel.get('source_entity') in top_entity_names or rel.get('target_entity') in top_entity_names:
                            filtered_relationships.append(rel)

                # If we found any top entities, create a result object for this document
                if top_entities:
                    final_results.append({
                        "id": hit.get('_id'),
                        "chunk_text": source.get('chunk_text'),
                        "entities": top_entities,
                        "relationships": filtered_relationships,
                        "score": hit.get('_score'), # This is the parent document's score
                        "file_name": metadata_from_hit.get('file_name'),
                        "doc_id": metadata_from_hit.get('doc_id'),
                        "page_number": metadata_from_hit.get('page_number'),
                        "chunk_index_in_page": metadata_from_hit.get('chunk_index_in_page')
                    })
            
            print(f"Semantic KG search found and processed {len(final_results)} documents.")
            return final_results
        except TransportError as e:
            print(f"Elasticsearch semantic KG search error: {e}", exc_info=True)
            return []
    
    async def _unified_rrf_search(self, query_text: str, query_embedding: List[float], top_k: int, top_k_entities: int, k_rrf: int = 60) -> List[Dict[str, Any]]:
        """
        Performs a unified RRF search across four sources: vector/keyword on chunks and vector/keyword on KG.
        """
        print(f"Performing Unified RRF Search for: '{query_text}'")
        tasks = [
            self._semantic_search_chunks(query_embedding, top_k),
            self._keyword_search_chunks(query_text, top_k),
        ]
        if top_k_entities > 0:
            tasks.extend([
                self._structured_kg_search(query_embedding, top_k, top_k_entities),
                self._keyword_search_kg(query_text, top_k)
            ])
        results = await asyncio.gather(*tasks, return_exceptions=True)

        if top_k_entities > 0:
            vec_chunks, key_chunks, vec_kg, key_kg = [res if isinstance(res, list) else [] for res in results]
        else:
            vec_chunks, key_chunks = [res if isinstance(res, list) else [] for res in results]
            vec_kg, key_kg = [], []
                
        ranks = {
            'vec_chunks': {doc['id']: i + 1 for i, doc in enumerate(vec_chunks)},
            'key_chunks': {doc['id']: i + 1 for i, doc in enumerate(key_chunks)},
            'vec_kg': {doc['id']: i + 1 for i, doc in enumerate(vec_kg)},
            'key_kg': {doc['id']: i + 1 for i, doc in enumerate(key_kg)}
        }

        all_docs = {}
        for doc in vec_chunks + key_chunks + vec_kg + key_kg:
            if doc['id'] not in all_docs or 'entities' in doc:
                all_docs[doc['id']] = doc

        for doc_id, doc in all_docs.items():
            score = 0.0
            for rank_list in ranks.values():
                if doc_id in rank_list:
                    score += 1.0 / (k_rrf + rank_list[doc_id])
            doc['score'] = score
        
        sorted_docs = sorted(all_docs.values(), key=lambda x: x['score'], reverse=True)
        print(f"Unified RRF search fused {len(sorted_docs)} unique documents.")
        return sorted_docs
    
    async def _keyword_search_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a precise keyword (phrase) search on the 'chunk_text' field.
        """
        if not query:
            print("Keyword search skipped: No query provided.")
            return []

        keyword_query = {
            "match_phrase": {
                "chunk_text": {
                    "query": query
                }
            }
        }
        index_name = self.config.get('index_name')
        print(f"Performing keyword (match_phrase) search with top_k={top_k}. Query: {query}")

        try:
            response = await self.es_client.search(
                index=index_name,
                query=keyword_query,
                size=top_k,
                _source_includes=["chunk_text", "metadata.file_name", "metadata.doc_id", "metadata.page_number", "metadata.chunk_index_in_page"]
            )
            results = []
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata = source.get('metadata', {})
                results.append({
                    "id": hit.get('_id'),
                    "text": source.get('chunk_text'),
                    "score": hit.get('_score'),
                    "file_name": metadata.get('file_name'),
                    "doc_id": metadata.get('doc_id'),
                    "page_number": metadata.get('page_number'),
                    "chunk_index_in_page": metadata.get('chunk_index_in_page')
                })
            print(f"Keyword search found {len(results)} chunks.")
            return results
        except TransportError as e:
            print(f"Elasticsearch keyword search error: {e}")
            return []
    
    async def _keyword_search_kg(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a keyword search and retrieves full knowledge graph data.
        """
        if not query:
            return []

        keyword_query = {"match": {"chunk_text": {"query": query, "fuzziness": "AUTO"}}}
        index_name = self.config.get('index_name')
        print(f"Performing KG keyword search with top_k={top_k}. Query: {query}")

        try:
            response = await self.es_client.search(
                index=index_name,
                query=keyword_query,
                size=top_k,
                _source_includes=["chunk_text", "metadata"]
            )
            results = []
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata = source.get('metadata', {})
                results.append({
                    "id": hit.get('_id'),
                    "chunk_text": source.get('chunk_text'),
                    "entities": metadata.get('entities', []),
                    "relationships": metadata.get('relationships', []),
                    "score": hit.get('_score'),
                    "file_name": metadata.get('file_name'),
                    "doc_id": metadata.get('doc_id'),
                    "page_number": metadata.get('page_number'),
                    "chunk_index_in_page": metadata.get('chunk_index_in_page')
                })
            return results
        except TransportError as e:
            print(f"Elasticsearch KG keyword search error: {e}")
            return []

    async def _rerank_documents(self, query: str, documents: List[Dict[str, Any]], doc_type: str, absolute_score_floor: float = 0.3) -> List[Dict[str, Any]]:
        if not documents:
            print(f"No {doc_type} documents to rerank for query: '{query[:50]}...'")
            return []
        if not self.reranker:
            print(f"Reranker not initialized. Skipping reranking for {doc_type} documents.")
            for doc in documents:
                if 'rerank_score' not in doc:
                    doc['rerank_score'] = doc.get('score') 
            return documents

        print(f"Reranking {len(documents)} {doc_type} documents for query: '{query[:50]}...'")

        if doc_type == "chunk":
            pairs = [(query, doc.get("text", "")) for doc in documents]
            text_key_for_logging = "text"
        elif doc_type == "kg":
            pairs = [(query, doc.get("chunk_text", "")) for doc in documents]
            text_key_for_logging = "chunk_text"
        else:
            print(f"Unknown document type '{doc_type}' for reranking. Skipping.")
            return documents

        try:
            scores = await asyncio.to_thread(self.reranker.predict, pairs, batch_size=8,activation_fct=torch.nn.Sigmoid())
            print(f"Successfully got scores for {len(pairs)} pairs for {doc_type} reranking.")
        except Exception as e:
            print(f"Error during reranking {doc_type} documents with CrossEncoder: {e}")
            # If reranking fails, ensure 'rerank_score' is present, set to original score or None
            for doc in documents:
                if 'rerank_score' not in doc:
                    doc['rerank_score'] = doc.get('score')
            return documents

        docs_with_scores = [{'doc': doc, 'score': scores[i]} for i, doc in enumerate(documents)]
        docs_with_scores.sort(key=lambda x: x['score'], reverse=True)

        # 1. Print all document scores in descending order
        print(f"\n--- Initial Reranked Scores for all {len(docs_with_scores)} documents ---")
        for i, item in enumerate(docs_with_scores):
            content_to_log = item['doc'].get(text_key_for_logging, '')
            print(f"  Rank {i+1}: Score={item['score']:.4f} | Content: '{content_to_log[:70]}...'")
        print("--- End of Initial Scores ---\n")

       # --- MODIFIED LOGIC: APPLY ABSOLUTE FLOOR FIRST ---
        # 1. Apply Absolute Score Cutoff as a primary quality gate.
        docs_passing_floor = [
            item for item in docs_with_scores if item['score'] >= absolute_score_floor
        ]
        print(f"Applied absolute score floor of {absolute_score_floor}. {len(docs_passing_floor)} of {len(docs_with_scores)} documents passed the quality gate.")

        if not docs_passing_floor:
            print("No documents met the absolute score floor. Returning empty list.")
            return []

        # --- HYBRID ELBOW METHOD LOGIC on the filtered set ---
        if len(docs_passing_floor) <= 2:
            print("Fewer than 3 documents passed the floor, returning all of them.")
            final_docs_with_scores = docs_passing_floor
        else:
            sorted_scores = [item['score'] for item in docs_passing_floor]
            
            score_diffs = [
                (sorted_scores[i] - sorted_scores[i+1]) / (sorted_scores[i] + 1e-9)
                for i in range(len(sorted_scores) - 1)
            ]

            elbow_index = 0
            if score_diffs:
                elbow_index = np.argmax(score_diffs)
            
            if score_diffs:
                elbow_doc = docs_passing_floor[elbow_index]
                elbow_score = elbow_doc['score']
                next_score = docs_passing_floor[elbow_index + 1]['score'] if (elbow_index + 1) < len(docs_passing_floor) else -1
                elbow_content = elbow_doc['doc'].get(text_key_for_logging, '')
                print(f"Elbow point detected after document at Rank {elbow_index + 1} (among docs that passed the floor).")
                print(f"  - Elbow Doc Score: {elbow_score:.4f} -> Next Doc Score: {next_score:.4f}")
                print(f"  - Elbow Doc Content: '{elbow_content[:70]}...'")

            num_to_keep = elbow_index + 1
            
            min_docs_to_keep = 3
            if num_to_keep < min_docs_to_keep and len(docs_passing_floor) >= min_docs_to_keep:
                print(f"Elbow method suggested keeping {num_to_keep}, but minimum is {min_docs_to_keep}. Adjusting to keep top {min_docs_to_keep}.")
                num_to_keep = min_docs_to_keep
            
            print(f"Dynamically selecting top {num_to_keep} documents after elbow analysis.")
            final_docs_with_scores = docs_passing_floor[:num_to_keep]
        
        reranked_docs = []
        print(f"\n--- Final Selected Documents (Top {len(final_docs_with_scores)} selected by hybrid method) ---")
        for i, item in enumerate(final_docs_with_scores):
            doc = item['doc']
            score = float(item['score'])
            doc['rerank_score'] = score
            content_to_log = doc.get(text_key_for_logging, '')
            print(f"  Final Rank {i+1}: Score={doc['rerank_score']:.4f} | Source: {doc.get('file_name', 'N/A')}, Page: {doc.get('page_number', 'N/A')} | Content: '{content_to_log[:70]}...'")
            reranked_docs.append(doc)
        print("--- End of Final Selected Documents ---\n")

        print(f"Successfully reranked and selected {len(reranked_docs)} {doc_type} documents using the hybrid elbow method after applying score floor.")
        return reranked_docs

    async def _perform_semantic_search_for_subquery(self, subquery_text: str, top_k: int) -> List[Dict[str, Any]]:
        print(f"Performing semantic search for subquery: '{subquery_text}'")
        embedding_list = await self._generate_embedding([subquery_text])
        if not embedding_list or not embedding_list[0]:
            print(f"Could not generate embedding for subquery: '{subquery_text}'. Semantic search will yield no results.")
            return []
        query_embedding= embedding_list[0]  # Get the first embedding if multiple texts were passed
        return await self._semantic_search_chunks(query_embedding, top_k)

    async def _perform_kg_search_for_subquery(self, subquery_text: str, top_k: int, top_k_entities:int) -> List[Dict[str, Any]]:
        print(f"Performing KG search for subquery: '{subquery_text}'")
        embedding_list= await self._generate_embedding([subquery_text])
        if not embedding_list or not embedding_list[0]:
            print(f"Could not generate embedding for subquery: '{subquery_text}'. KG search will yield no results.")
            return []
        query_embedding = embedding_list[0]  # Get the first embedding if multiple texts were passed
        return await self._structured_kg_search(query_embedding, top_k, top_k_entities)

    def _generate_shorthand_id(self, item: Dict[str, Any], prefix: str, index: int) -> str:
        doc_id_part = "unknown"
        if item.get("doc_id"):
            doc_id_part = str(item["doc_id"]).replace('-', '')[:6]
        
        page_num_val = item.get("page_number")
        page_num_part = str(page_num_val) if page_num_val is not None else "NA"
        
        chunk_idx_val = item.get("chunk_index_in_page")
        chunk_idx_part = str(chunk_idx_val) if chunk_idx_val is not None else str(index)
        
        return f"{prefix}_{doc_id_part}_p{page_num_part}_i{chunk_idx_part}"

    def _format_search_results_for_llm(self, original_query: str, sub_queries_results: List[Dict[str, Any]]) -> str:
        lines = [f"Original Query: {original_query}\n"]
        
        if not sub_queries_results:
            lines.append("No search results found.")
            return "\n".join(lines)

        for sq_idx, sq_result in enumerate(sub_queries_results):
            sub_query_text = sq_result.get("sub_query_text", f"Sub-query {sq_idx + 1}")
            lines.append(f"--- Results for Sub-query: \"{sub_query_text}\" ---")

            reranked_chunks = sq_result.get("reranked_chunks", [])
            if reranked_chunks:
                lines.append("\nVector Search Results (Chunks):")
                for chunk_idx, chunk in enumerate(reranked_chunks):
                    if not isinstance(chunk, dict):
                        print(f"Skipping non-dict chunk item during formatting: {chunk}")
                        continue

                    shorthand_id = self._generate_shorthand_id(chunk, "c", chunk_idx)
                    score_val = chunk.get('rerank_score', chunk.get('score'))
                    score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                    lines.append(f"Source ID [{shorthand_id}]: (Score: {score_str})")
                    
                    text_content = chunk.get("text") or chunk.get("chunk_text", "N/A")
                    lines.append(text_content)
                    lines.append(f"  File: {chunk.get('file_name', 'N/A')}, Page: {chunk.get('page_number', 'N/A')}, Chunk Index in Page: {chunk.get('chunk_index_in_page', 'N/A')}")
            else:
                lines.append("\nNo vector search results for this sub-query.")

            retrieved_kg_data = sq_result.get("retrieved_kg_data", []) # This is `final_kg_evidence_for_output`
            if retrieved_kg_data:
                lines.append("\nKnowledge Graph Results:")
                for kg_idx, kg_item in enumerate(retrieved_kg_data):
                    if not isinstance(kg_item, dict):
                        print(f"Skipping non-dict kg_item during formatting: {kg_item}")
                        continue
                    
                    shorthand_id = self._generate_shorthand_id(kg_item, "kg", kg_idx)
                    score_val = kg_item.get('rerank_score', kg_item.get('score')) # KG items might also have rerank_score
                    score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                    lines.append(f"Source ID [{shorthand_id}]: (Score: {score_str})")
                    lines.append(f"  File: {kg_item.get('file_name', 'N/A')}, Page: {kg_item.get('page_number', 'N/A')}, Chunk Index in Page: {kg_item.get('chunk_index_in_page', 'N/A')}")

                    entities = kg_item.get("entities", [])
                    if entities:
                        lines.append("  Entities:")
                        for entity in entities:
                            if not isinstance(entity, dict): continue
                            lines.append(f"    - Name: {entity.get('name', 'N/A')}, Type: {entity.get('type', 'N/A')}")
                            entity_desc = entity.get('description', '')
                            if entity_desc:
                                lines.append(f"      Description: {entity_desc}")
                    
                    relationships = kg_item.get("relationships", [])
                    if relationships:
                        lines.append("  Relationships:")
                        for rel in relationships:
                            if not isinstance(rel, dict): continue
                            lines.append(f"    - {rel.get('source_entity', 'S')} -> {rel.get('relation', 'R')} -> {rel.get('target_entity', 'T')} (Weight: {rel.get('relationship_weight', 'N/A')})")
                            rel_desc=rel.get("relationship_description", "")
                            if rel_desc:
                                lines.append(f"    Description:{rel_desc}")
            
            else:
                lines.append("\nNo knowledge graph results for this sub-query.")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def _extract_keywords_for_search(self, user_query: str, schema_chunks: str) -> List[str]:
        """
        Uses an LLM to extract the single most relevant keyword from the user query,
        using a fetched sample document for schema context.
        """
        
        query_words = set(word.lower() for word in user_query.replace('.', ' ').replace(',', ' ').split())
        
        system_prompt = """You are an expert at information retrieval and search query optimization. 
Your task is to analyze a user's query and extract the SINGLE most essential keyword FROM THE USER'S QUERY ITSELF.
The keyword you select MUST appear in the user's original query - this is a strict requirement."""
            
        context_lines = []
        for fname, chunk in schema_chunks.items():
            context_lines.append(f"File: {fname}\nSample Chunk: {chunk}\n---")
        schema_context = "\n".join(context_lines)
        
        user_prompt = """
I need you to extract the SINGLE most important keyword from the user's query below. This keyword will be used for database search filtering.

**IMPORTANT RULES:**
1. You MUST select a keyword that appears VERBATIM in the user's query.
2. DO NOT invent or suggest terms that aren't explicitly in the query.
3. Ignore any terms that might be in the domain context but not in the query.
4. Choose the most specific and relevant term from the query.

**Priority order for selection:**
1. Specific named entities (names, IDs, unique terms)
2. Domain-specific technical terms 
3. Important nouns relevant to the question's subject
4. Descriptive adjectives if they narrow the search
5. Action verbs as a last resort

**Domain Context (For understanding ONLY - do not extract terms from here):**
{schema_context}

**User Query:** "{user_query}"

**Your Response:**
Return ONLY the single chosen keyword, with no explanations, quotes, or additional text.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(schema_context=schema_context, user_query=user_query)}
        ]
        
        llm_response_content = ""
        print(f"Extracting keyword from query via LLM: '{user_query}'")
        llm_response_content = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL, payload_messages=messages, max_tokens=50, temperature=0.0
        )

        # if not llm_response_content:
        #     print("⚠️ LLM returned no keywords. Falling back to using the original query.")
        #     return [user_query]
        
        if not llm_response_content:
            print("⚠️ LLM returned no keywords. Falling back to using the most substantial noun in query.")
        # Choose the longest word as fallback (simple heuristic)
            words = [w for w in user_query.split() if len(w) > 3]
            fallback_keyword = max(words, key=len) if words else user_query
            return [fallback_keyword]

        keyword = llm_response_content.strip().lower()
        
        if keyword in user_query.lower():
            print(f"✅ Extracted keyword: '{keyword}'")
            return [keyword]
        else:
            if keyword in query_words:
                print(f"✅ Extracted keyword (word match): '{keyword}'")
                return [keyword]
            else:
                print(f"⚠️ Extracted keyword '{keyword}' not found in query. Using most specific noun instead.")
            # Get the longest noun from the query as fallback
                nouns = [word for word in user_query.split() if len(word) > 3 and word.lower() not in ['list', 'what', 'when', 'where', 'how', 'who', 'which']]
                best_keyword = max(nouns, key=len) if nouns else "vessel" if "vessel" in user_query.lower() else user_query.split()[-1]
                return [best_keyword]
    
    async def _determine_optimal_chunk_count(self, user_query: str, query_type: str = None) -> int:
        """
        Dynamically determines the optimal number of chunks to retrieve based on:
        1. Query type and complexity
        2. Database size and characteristics 
        3. System resource constraints
        """
        # Default fallback values if analysis fails
        default_counts = {
            'factual_lookup': 5,
            'summary_extraction': 10,
            'comparison': 8,
            'complex_analysis': 15
        }
        
        try:
            # Get query type if not provided
            if not query_type:
                query_type = await self._classify_query_type(user_query)
            
            # Step 1: Analyze database size
            index_name = self.config.get('index_name')
            index_stats = await self.es_client.count(index=index_name)
            total_docs = index_stats.get('count', 1000)  # Default assumption if count fails
            
            print(f"Database size analysis: {total_docs} total documents in index")
            
            # Step 2: Calculate base chunk count based on query type
            base_count = default_counts.get(query_type, 8)
            
            # Step 3: Adjust for database size
            if total_docs < 1000:
                # For small databases, retrieve a higher percentage
                size_adjusted_count = max(base_count, int(total_docs * 0.05))
            elif total_docs < 10000:
                # For medium databases
                size_adjusted_count = max(base_count, int(total_docs * 0.02))
            else:
                # For large databases
                size_adjusted_count = max(base_count, int(total_docs * 0.01))
                
            # Step 4: Apply query complexity adjustments
            complexity_factor = 1.0
            
            # Check for indicators of complex queries
            if len(user_query.split()) > 15:
                complexity_factor *= 1.3  # Longer queries may need more context
            
            if "compare" in user_query.lower() or "difference" in user_query.lower():
                complexity_factor *= 1.2  # Comparative queries need more context
                
            if "list" in user_query.lower() or "all" in user_query.lower():
                complexity_factor *= 1.4  # Queries asking for comprehensive lists
                
            # LLM-based complexity analysis (lightweight version)
            try:
                system_prompt = "You are an AI that evaluates query complexity. Rate the following query on a scale from 1 to 10, where 1 is extremely simple and 10 is very complex. Return ONLY the number."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {user_query}\n\nComplexity rating (1-10):"}
                ]
                
                complexity_response = await self._call_openai_api(
                    model_name=OPENAI_CHAT_MODEL,
                    payload_messages=messages,
                    max_tokens=10,
                    temperature=0.1
                )
                
                try:
                    complexity_score = float(complexity_response.strip())
                    if 1 <= complexity_score <= 10:
                        # Adjust complexity factor based on LLM score (1.0 to 2.0)
                        complexity_factor *= (1.0 + (complexity_score - 1) / 9)
                except (ValueError, TypeError):
                    pass  # If conversion fails, keep existing complexity factor
                    
            except Exception as e:
                print(f"LLM-based complexity analysis failed: {e}")
                # Continue with existing complexity factor
                
            # Apply complexity factor
            final_count = int(size_adjusted_count * complexity_factor)
            
            # Step 5: Apply practical bounds
            min_chunks = 3  # Minimum chunks to ensure we get some context
            max_chunks = 30  # Maximum to prevent excessive resource usage
            final_count = max(min_chunks, min(final_count, max_chunks))
            
            # Step 6: Account for resource constraints
            resource_limit = self.params.get('max_chunks', 50)
            final_count = min(final_count, resource_limit)
            
            print(f"Dynamic chunk count determination: {final_count} chunks for query type '{query_type}' (complexity factor: {complexity_factor:.2f})")
            return final_count
            
        except Exception as e:
            print(f"Error in optimal chunk count determination: {e}. Using default value.")
            return default_counts.get(query_type, 10)
        
    async def _generate_final_answer(self, original_query: str, llm_formatted_context: str, cited_files: List[str]) -> str:
        """Generates the final answer by sending the context to the appropriate LLM."""
        print("\n--- Generating Final Answer based on Synthesized Context ---")
        
        if not llm_formatted_context or not llm_formatted_context.strip():
            print("⚠️ Cannot generate final answer: Formatted context is empty.")
            return "Could not generate an answer because no relevant information was found."
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
        original_query=original_query,
        context=llm_formatted_context
        )

        messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": user_prompt}
    ]
        max_tokens = 16000

        final_answer = ""
        print("Default mode: Routing final answer generation to OpenAI API.")
        final_answer = await self._call_openai_api(
                model_name=OPENAI_CHAT_MODEL,
                payload_messages=messages,
                max_tokens=max_tokens,
                temperature=0.3
            )
        
        if final_answer:
            print("✅ Successfully generated final answer.")
            return final_answer
        else:
            error_msg = f"The {OPENAI_CHAT_MODEL} model returned an empty response."
            print(f"❌ {error_msg}")
            return error_msg

    async def search(
        self,
        user_query: str,
        num_subqueries: int = 2,
        initial_candidate_pool_size: int = 50,
        top_k_kg_entities: int = 8,
        absolute_score_floor: float = 0.3
    ) -> Dict[str, Any]:
        print(f"Starting RAG Fusion search for user query: '{user_query}'")

        schema_chunks = await self._fetch_schema_chunks_by_file()
        query_type = await self._classify_query_type(user_query)
        
        all_retrieved_chunks: Dict[str, Any] = {}
        
        if query_type == 'factual_lookup':
            print("-> Factual Lookup: Using single keyword search.")
            search_keywords = await self._extract_keywords_for_search(user_query, schema_chunks)
            
            # For factual lookups, perform a precise keyword search
            if search_keywords:
                retrieved_chunks_list = await self._keyword_search_chunks(search_keywords[0], initial_candidate_pool_size)
                for chunk in retrieved_chunks_list:
                    all_retrieved_chunks[chunk['id']] = chunk
        else:
            print("-> Complex Query: Generating subqueries and running RRF for each.")
            subqueries = await self._generate_subqueries(user_query, num_subqueries=num_subqueries)
            subqueries.insert(0, user_query) # Also process the original query

            for sq_text in subqueries:
                print(f"\n--- Processing Complex Subquery: '{sq_text}' ---")
                
                query_embeddings = await self._generate_embedding([sq_text])
                if not query_embeddings or not query_embeddings[0]:
                    continue
                query_embedding = query_embeddings[0]

                # Extract the single most important keyword from the subquery
                extracted_keywords = await self._extract_keywords_for_search(sq_text, schema_chunks)
                if not extracted_keywords:
                    continue
                single_keyword = extracted_keywords[0]

                # Perform RRF using the full text for vector and the single keyword for keyword search
                fused_chunks = await self._unified_rrf_search(
                    query_text=single_keyword,
                    query_embedding=query_embedding,
                    top_k=initial_candidate_pool_size,
                    top_k_entities=top_k_kg_entities
                )
                
                for chunk in fused_chunks:
                    if chunk['id'] not in all_retrieved_chunks:
                        all_retrieved_chunks[chunk['id']] = chunk

        retrieved_chunks = list(all_retrieved_chunks.values())
        
        retrieved_kg_evidence_with_chunk_text = []
        original_query_embedding = await self._generate_embedding([user_query])
        if original_query_embedding and original_query_embedding[0]:
             retrieved_kg_evidence_with_chunk_text = await self._structured_kg_search(
                 original_query_embedding[0], initial_candidate_pool_size, top_k_kg_entities
             ) 
                
        if self.reranker and self.deep_research:
            print("Deep research is ON. Reranking and pruning will be applied to the final candidate pool.")
            retrieved_chunks = await self._rerank_documents(user_query, retrieved_chunks, "chunk", absolute_score_floor)
            if self.provence_pruner:
                retrieved_chunks = await self._prune_documents(user_query, retrieved_chunks, "chunk")

            retrieved_kg_evidence_with_chunk_text = await self._rerank_documents(user_query, retrieved_kg_evidence_with_chunk_text, "kg", absolute_score_floor)
            if self.provence_pruner:
                 retrieved_kg_evidence_with_chunk_text = await self._prune_documents(user_query, retrieved_kg_evidence_with_chunk_text, "kg")

        final_kg_evidence_for_output = []
        for doc in retrieved_kg_evidence_with_chunk_text:
            doc_copy = doc.copy()
            doc_copy.pop("chunk_text", None)
            final_kg_evidence_for_output.append(doc_copy)
            
        processed_subquery_results = [{
                    "sub_query_text": user_query,
                    "reranked_chunks": retrieved_chunks,
                    "retrieved_kg_data": final_kg_evidence_for_output
                }]
        
        final_results_dict = {
            "original_query": user_query,
            "sub_queries_results": processed_subquery_results,
            # "refrences": citations_str
        }
        
        llm_formatted_context = self._format_search_results_for_llm(
            original_query=user_query,
            sub_queries_results=processed_subquery_results 
        )
        final_results_dict["llm_formatted_context"] = llm_formatted_context
        
        show_citations = self.params.get('enable_references_citations', False)
        # show_references = self.params.get('reference', False)
        

        if show_citations:
            cited_files = set()
            referenced_chunk_ids = set()

            for sq_result in processed_subquery_results:
                for chunk in sq_result.get("reranked_chunks", []):
                    if chunk.get("file_name"):
                        cited_files.add(chunk["file_name"])
                    # if chunk.get("id"):
                    #     referenced_chunk_ids.add(chunk["id"])
                
                for kg_item in sq_result.get("retrieved_kg_data", []):
                    if kg_item.get("file_name"):
                        cited_files.add(kg_item["file_name"])
                    # if kg_item.get("id"):
                    #     referenced_chunk_ids.add(kg_item["id"])
            
            final_answer = await self._generate_final_answer(
            original_query=user_query,
            llm_formatted_context=llm_formatted_context,
            cited_files=list(cited_files),
        )
        
        final_results_dict["final_answer"] = final_answer
        
        print("\n--- Final Answer ---")
        print(f'{final_answer}', "No optimized context generated.")

        print(f"RAG Fusion search fully completed for user query: '{user_query}'. Formatted context and final answer generated.")
        return final_results_dict

async def handle_request(data: Message) -> FunctionResponse:
  es_client: Optional[AsyncElasticsearch] = None
  aclient_openai: Optional[AsyncOpenAI] = None
  
  try:
    print('Incoming Data:--', data)
    params = data.params
    config = data.config

    es_client = await check_async_elasticsearch_connection()
    if not es_client:
      return FunctionResponse(False, "Could not connect to Elasticsearch.")
    
    aclient_openai = await init_async_openai_client()
    if not aclient_openai:
        return FunctionResponse(False, "Could not connect to Open Ai.")

    retriever = RAGFusionRetriever(params, config, es_client, aclient_openai)
    user_query_input = params.get('question')
    
    query_type = await retriever._classify_query_type(user_query_input)

    if params.get('auto_chunk_sizing', True):  # Allow override with a flag
        top_k_chunks = await retriever._determine_optimal_chunk_count(user_query_input, query_type)
        print(f"Determined optimal chunk count: {top_k_chunks} for query type '{query_type}'")
    else:
        # Use provided value or default
        top_k_chunks = int(params.get('top_k_chunks', 6))
    
    print(f"\n--- Running RAG Fusion Search for: '{user_query_input}' ---")
    search_results_dict = await retriever.search(
        user_query=user_query_input, initial_candidate_pool_size=top_k_chunks, top_k_kg_entities=top_k_chunks, absolute_score_floor=0.3
    )
    print("\n--- Search Results Dictionary (RAG Fusion: Chunks & KG Reranked if applicable) ---")
    
    # print("\n--- LLM Formatted Context ---")
    # print(search_results_dict.get("llm_formatted_context", "No formatted context generated."))

    if es_client and hasattr(es_client, 'close'):
      await es_client.close()
      print("Elasticsearch client closed.")
    if aclient_openai and hasattr(aclient_openai, "aclose"):
        print('open ai clinet a close')
        try:
          await aclient_openai.aclose()
          print("OpenAI client closed.")
        except Exception as e:
          print(f"Error closing OpenAI client: {e}")

    return FunctionResponse(message=Messages(search_results_dict.get("llm_formatted_context", "No formatted context generated.")), failed=False) 
  except Exception as e:
    print(f"❌ Error during retrieval: {e}")
    return FunctionResponse(message=Messages(e))

def test_query():
    params = {
        "question": "List the main characteristics of the vessel",
        "top_k_chunks": 1,
        "enable_references_citations": True,
        "deep_research": False,
    }
    config = {
        "index_name": "hydrostatic",
    }
    message = Message(params=params, config=config)
    res = asyncio.run(handle_request(message))
    # print('res of handle request:-', res)

if __name__ == "__main__":
    test_query()
