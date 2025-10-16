<div align='center'>
  <img src="app/assets/cover.png" alt="RowBlaze Logo" width="800"/>
  <br>
  <b>Most accurate Multimodal Agentic RAG for Structured and Unstructured data</b>

</div>

<div align='center'>
<a href="https://github.com/negativenagesh/RowBlaze/stargazers">
    <img src="https://img.shields.io/github/stars/negativenagesh/RowBlaze?style=flat&logo=github" alt="Stars">
  </a>
  <a href="https://github.com/negativenagesh/RowBlaze/network/members">
    <img src="https://img.shields.io/github/forks/negativenagesh/RowBlaze?style=flat&logo=github" alt="Forks">
  </a>
  <a href="https://github.com/negativenagesh/RowBlaze/pulls">
    <img src="https://img.shields.io/github/issues-pr/negativenagesh/RowBlaze?style=flat&logo=github" alt="Pull Requests">
  </a>
  <a href="https://github.com/negativenagesh/RowBlaze/issues">
    <img src="https://img.shields.io/github/issues/negativenagesh/RowBlaze?style=flat&logo=github" alt="Issues">
  </a>
  <a href="https://github.com/negativenagesh/RowBlaze/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/negativenagesh/RowBlaze?style=flat&logo=github" alt="License">
  </a>

</div>

<br></br>

## Signup/in, functionalities/available options

<div align="center">

https://github.com/user-attachments/assets/afe00a07-0344-463a-bbb5-071e3d7f5e70

</div>

## Agentic RAG Retrieval

<div align="center">

https://github.com/user-attachments/assets/2c6d6aa0-139f-4f53-b54a-96d709064289

</div>

## Setup locally

### 1. Clone and star the Repository

```bash
git clone https://github.com/negativenagesh/RowBlaze.git
cd RowBlaze
```

---

### 2. Install Dependencies

```bash
uv init
uv venv
uv pip install -r requirements.txt
```

---

### 3. Environment Variables

Create a `.env` file in the root directory.
Below is an example of the required variables:

```env
# OpenAI API
OPEN_AI_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_SUMMARY_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Elasticsearch
RAG_UPLOAD_ELASTIC_URL=your_elasticsearch_URL_from_elasticsearch_cloud
ELASTICSEARCH_API_KEY=your_elasticsearch_api_key

# (Optional) Local reranker model
RERANKER_MODEL_ID=BAAI/bge-reranker-base
```

---

#### For API/Core Usage

You can run the ingestion or retrieval modules directly for testing:

```bash
uv run src/core/ingestion/rag_ingestion.py
uv run src/core/retrieval/rag_retrieval.py
```

---

#### For the App

```bash
streamlit run app/app.py
```

---

### 7. Notes

- Ensure your Elasticsearch instance is running and accessible.
- Your OpenAI API key must have access to the specified models.
- For OCR, install `pytesseract` and ensure Tesseract is in your system PATH.
- For local reranking, download the model specified by `RERANKER_MODEL_ID` if not using the default.

---

### 8. Docker Setup (Recommended for Production)

RowBlaze provides a complete Docker setup with separate containers for the API, Streamlit app, and nginx reverse proxy.

#### Prerequisites

- Docker and Docker Compose installed
- `.env` file configured (see step 3 above)

#### Quick Start with Docker

1. **Clone the repository** (if not already done):

```bash
git clone https://github.com/negativenagesh/RowBlaze.git
cd RowBlaze
```

2. **Create your `.env` file** with the required environment variables (see step 3 above).

3. **Build and run with Docker Compose**:

```bash
docker-compose up --build
```

This will start:

- **API service** on port 8000 (internal)
- **Streamlit app** on port 8501 (internal)
- **Nginx reverse proxy** on port 80 (external access point)

4. **Access the application**:
   - Open your browser and go to `http://localhost`
   - The nginx proxy will route requests appropriately

#### Individual Container Setup

If you prefer to run containers individually:

**Build the API container:**

```bash
docker build -f Dockerfile.api -t rowblaze-api .
docker run -p 8000:8000 --env-file .env rowblaze-api
```

**Build the App container:**

```bash
docker build -f Dockerfile.app -t rowblaze-app .
docker run -p 8501:8501 --env-file .env rowblaze-app
```

**Build the Nginx container:**

```bash
docker build -f nginx/Dockerfile -t rowblaze-nginx ./nginx
docker run -p 80:80 rowblaze-nginx
```

#### Docker Compose Configuration

The `docker-compose.yml` includes:

- **API service**: FastAPI backend with health checks
- **App service**: Streamlit frontend with health checks
- **Nginx service**: Reverse proxy for routing and load balancing
- **Shared network**: For inter-service communication
- **Volume mounts**: For persistent data and uploads

#### Environment Variables for Docker

Ensure your `.env` file includes:

```env
# API Configuration
ROWBLAZE_API_URL=http://api:8000/api

# OpenAI API
OPEN_AI_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini

# Elasticsearch
RAG_UPLOAD_ELASTIC_URL=your_elasticsearch_URL
ELASTICSEARCH_API_KEY=your_elasticsearch_api_key

# Authentication
JWT_SECRET_KEY=your-secret-key-change-in-production
```

#### Health Checks

Both containers include health checks:

- **API**: `GET /api/health`
- **App**: Streamlit's built-in health endpoint
- **Nginx**: Basic connectivity check

#### Troubleshooting Docker Setup

**Container logs:**

```bash
docker-compose logs api
docker-compose logs app
docker-compose logs nginx
```

**Restart services:**

```bash
docker-compose restart
```

**Rebuild after changes:**

```bash
docker-compose down
docker-compose up --build
```

**Check container status:**

```bash
docker-compose ps
```

---

### 9. Demo

See the demo links and screenshots at the top of this README for usage examples.

---

## Agentic RAG Architecture

RowBlaze implements a sophisticated **Agentic RAG** system that goes beyond traditional RAG by using an intelligent agent that can reason about queries, select appropriate tools, and iteratively gather information to provide comprehensive answers.

### How Agentic RAG Works

The Agentic RAG system is built around the `StaticResearchAgent` class, which acts as an intelligent coordinator that:

1. **Analyzes user queries** using GPT-4o-mini to classify intent and complexity
2. **Selects optimal tools** based on query requirements and available capabilities
3. **Executes multi-step research** through iterative tool usage
4. **Synthesizes comprehensive answers** from gathered information across multiple sources

### Core Components

#### 1. StaticResearchAgent (Brain of the System)

The agent (`src/core/agents/static_research_agent.py`) serves as the central intelligence that:

- **Query Classification**: Uses LLM to categorize queries into:

  - `factual_lookup`: Simple, direct questions
  - `summary_extraction`: Requests for overviews/summaries
  - `comparison`: Comparative analysis between items
  - `complex_analysis`: Multi-faceted analytical questions

- **Dynamic Tool Selection**: Intelligently chooses 1-3 most relevant tools from available options based on query type and requirements

- **Iterative Research Process**: Executes up to 2 iterations of tool-based research, with GPT-4o-mini evaluating sufficiency after each iteration

- **Context Synthesis**: Combines results from multiple tools and iterations into coherent, comprehensive answers

#### 2. Specialized Research Tools

The agent has access to 4 specialized tools (`src/core/tools/`):

**🔍 VectorSearchTool** (`vectorsearch.py`)

- **Purpose**: Semantic similarity search using embeddings
- **Best For**: Conceptual queries, finding related content, thematic searches
- **How It Works**: Converts queries to embeddings and performs cosine similarity search against document embeddings

**🔤 KeywordSearchTool** (`KeywordSearch.py`)

- **Purpose**: Exact phrase and keyword matching
- **Best For**: Finding specific terms, names, quotes, or precise information
- **How It Works**: Uses Elasticsearch's match_phrase and fuzzy matching for precise retrieval

**📚 SearchFileKnowledgeTool** (`search_file_knowledge.py`)

- **Purpose**: Comprehensive document analysis and broad information gathering
- **Best For**: Summaries, overviews, and when extensive context is needed
- **How It Works**: Performs multi-subquery RAG fusion search with reranking

**🕸️ GraphTraversalTool** (in `static_research_agent.py`)

- **Purpose**: Entity relationship analysis and graph-based queries
- **Best For**: Questions about connections, relationships, and entity interactions
- **How It Works**: Generates Cypher-like queries and retrieves knowledge graph data

#### 3. Intelligent Decision Making

**Query Complexity Assessment**:

```python
# The agent uses GPT-4o-mini to determine optimal parameters
dynamic_top_k = await self._get_dynamic_top_k(query, query_type)
optimal_tools = await self._determine_optimal_tools(query, query_type)
```

**Sufficiency Evaluation**:

```python
# After each iteration, GPT-4o-mini evaluates if more information is needed
sufficiency_assessment = await self._enhanced_sufficiency_check(
    initial_context, query, query_type
)
```

### Agentic RAG Workflow

#### Phase 1: Query Analysis & Planning

1. **Query Classification**: LLM analyzes query intent and complexity
2. **Tool Selection**: Agent selects 1-3 optimal tools based on query requirements
3. **Parameter Optimization**: Dynamic determination of top-k values and search parameters

#### Phase 2: Information Gathering

1. **Initial Retrieval** (Optional): Traditional RAG search for baseline context
2. **Sufficiency Check**: GPT-4o-mini evaluates if initial results are adequate
3. **Tool Execution**: If insufficient, agent executes selected tools in parallel
4. **Iterative Research**: Up to 2 iterations with re-evaluation between iterations

#### Phase 3: Synthesis & Response

1. **Context Combination**: Merges results from all tools and iterations
2. **Answer Generation**: LLM synthesizes comprehensive response with citations
3. **Quality Assurance**: Ensures traceability and source attribution

### Operating Modes

#### Normal RAG Mode

- Performs initial retrieval using traditional RAG fusion
- Uses agent tools only if initial results are insufficient
- Balances speed with comprehensiveness

#### Pure Agentic Mode

- **No initial retrieval** - agent must use tools for ALL information gathering
- Forces complete tool-based research for maximum thoroughness
- Ideal for complex analytical queries requiring comprehensive coverage

```python
# Pure agentic mode configuration
agent_config = {
    "perform_initial_retrieval": False,  # Forces pure agentic behavior
    "max_iterations": 2,
    "tool_top_k_chunks": 20,
    "tool_top_k_kg": 10
}
```

### Advanced Features

#### 1. Parallel Tool Execution

Tools are executed concurrently for efficiency:

```python
tool_results = await self._execute_tool_calls_in_parallel(
    parsed_tool_calls, current_config, messages
)
```

#### 2. Smart Caching

- Response caching for similar queries
- Avoids redundant API calls and computations

#### 3. Error Handling & Fallbacks

- Graceful degradation when tools fail
- Automatic retry with exponential backoff
- Fallback to alternative search methods

#### 4. Resource Management

- Proper cleanup of async clients and connections
- Rate limiting to prevent API overuse
- Memory-efficient result processing

### Tool Selection Intelligence

The agent uses sophisticated logic to select optimal tools:

```python
# Example tool selection for different query types
if query_type == "factual_lookup":
    tools = ["vector_search", "keyword_search"]
elif query_type == "complex_analysis":
    tools = ["search_file_knowledge", "vector_search", "graph_traversal"]
elif query_type == "comparison":
    tools = ["search_file_knowledge", "vector_search", "keyword_search"]
```

### Performance Optimizations

1. **Dynamic Chunk Sizing**: Automatically adjusts retrieval parameters based on query complexity
2. **Intelligent Stopping**: GPT-4o-mini determines when sufficient information is gathered
3. **Selective Tool Usage**: Only uses necessary tools, avoiding over-retrieval
4. **Parallel Processing**: Concurrent tool execution reduces latency

### Integration with Traditional RAG

The agentic system seamlessly integrates with RowBlaze's traditional RAG pipeline:

- **Shared Infrastructure**: Uses same Elasticsearch indices and embedding models
- **Unified API**: Single endpoint serves both traditional and agentic RAG
- **Consistent Output**: Both modes provide structured responses with citations
- **Flexible Configuration**: Easy switching between modes based on use case

This agentic approach enables RowBlaze to handle complex, multi-faceted queries that would be challenging for traditional RAG systems, while maintaining the speed and efficiency of simpler retrieval for straightforward questions.

---

## Normal RAG Architecture

RowBlaze's **Normal RAG** system provides fast, efficient document retrieval and answer generation through a sophisticated multi-stage pipeline that combines semantic search, keyword matching, and knowledge graph retrieval.

### How Normal RAG Works

The Normal RAG system is built around the `RAGFusionRetriever` class in `src/core/retrieval/rag_retrieval.py`, which orchestrates a comprehensive retrieval and synthesis pipeline:

#### 1. Query Processing & Classification

**Query Analysis**:

- **Intent Classification**: Uses GPT-4o-mini to classify queries into categories:
  - `factual_lookup`: Specific information requests (dates, names, numbers)
  - `summary_extraction`: Document overviews and summaries
  - `comparison`: Comparative analysis between items
  - `complex_analysis`: Multi-faceted analytical questions

**Schema-Aware Processing**:

- **Schema Sampling**: Fetches representative chunks from each indexed file to understand data structure
- **Keyword Extraction**: Uses LLM to extract the most relevant search terms based on query and schema context
- **Dynamic Optimization**: Automatically determines optimal chunk count based on query complexity and database size

#### 2. Multi-Strategy Retrieval Pipeline

**RAG Fusion Approach**:

```python
# Generates multiple focused subqueries for comprehensive coverage
subqueries = await self._generate_subqueries(user_query, num_subqueries=2)
```

**Parallel Search Execution**:

- **Semantic Search**: Vector similarity using OpenAI embeddings (text-embedding-3-large, 3072 dimensions)
- **Keyword Search**: Exact phrase matching and fuzzy text search
- **Knowledge Graph Search**: Entity and relationship retrieval using embedded descriptions
- **Unified RRF Fusion**: Combines results using Reciprocal Rank Fusion for optimal diversity

#### 3. Advanced Search Techniques

**Vector Search (`_semantic_search_chunks`)**:

```python
knn_query = {
    "field": "embedding",
    "query_vector": query_embedding,
    "k": top_k,
    "num_candidates": top_k * 10,
}
```

- Uses cosine similarity for semantic matching
- Searches against chunk embeddings and entity description embeddings
- Handles multi-dimensional vector spaces efficiently

**Keyword Search (`_keyword_search_chunks`)**:

```python
keyword_query = {"match_phrase": {"chunk_text": {"query": query}}}
```

- Performs exact phrase matching for precision
- Uses Elasticsearch's fuzzy matching for typo tolerance
- Optimized for factual lookups and specific term searches

**Knowledge Graph Retrieval (`_structured_kg_search`)**:

- Searches entity embeddings using nested queries
- Retrieves related entities and relationships
- Provides structured context for complex queries

#### 4. Result Fusion & Ranking

**Reciprocal Rank Fusion (RRF)**:

```python
# Combines multiple search strategies with weighted scoring
score += 1.0 / (k_rrf + rank_list[doc_id])
```

**Advanced Reranking** (Optional):

- **Cross-Encoder Reranking**: Uses BAAI/bge-reranker-base for relevance scoring
- **Elbow Method Selection**: Dynamically determines optimal result count
- **Absolute Score Filtering**: Applies quality thresholds (default: 0.3)

**Content Pruning** (Optional):

- **Provence Pruning**: Removes irrelevant content while preserving context
- **Smart Truncation**: Focuses on most relevant passages

#### 5. Context Assembly & Formatting

**Structured Context Creation**:

```python
def _format_search_results_for_llm(self, original_query, sub_queries_results):
    # Formats retrieved chunks and KG data into structured context
    # Includes source attribution and metadata
```

**Citation Management**:

- Tracks all source files and page numbers
- Enforces paragraph-level citations
- Maintains traceability throughout the pipeline

#### 6. Answer Generation

**LLM Synthesis**:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
    {"role": "user", "content": user_prompt}
]
```

**System Prompt Engineering**:

- Enforces strict context-bound responses
- Requires structured formatting with markdown
- Mandates source citations for traceability
- Prevents hallucination through explicit constraints

### Normal RAG Workflow

#### Phase 1: Query Understanding

1. **Classification**: Determine query type and complexity
2. **Schema Analysis**: Sample database structure for context
3. **Keyword Extraction**: Identify optimal search terms
4. **Parameter Optimization**: Set dynamic chunk counts and search parameters

#### Phase 2: Multi-Modal Retrieval

1. **Subquery Generation**: Create focused search variations using RAG Fusion
2. **Parallel Search**: Execute semantic, keyword, and KG searches simultaneously
3. **Result Fusion**: Combine results using RRF for optimal coverage
4. **Quality Filtering**: Apply reranking and pruning if enabled

#### Phase 3: Context Synthesis

1. **Context Assembly**: Structure retrieved information with metadata
2. **Citation Tracking**: Maintain source attribution throughout
3. **Format Optimization**: Prepare context for LLM consumption

#### Phase 4: Answer Generation

1. **LLM Synthesis**: Generate comprehensive answers using structured prompts
2. **Citation Integration**: Embed source references at paragraph level
3. **Quality Assurance**: Ensure factual grounding and traceability

### Key Features of Normal RAG

#### 1. Intelligent Query Processing

- **Dynamic Chunk Sizing**: Automatically adjusts retrieval parameters based on query complexity
- **Schema-Aware Search**: Uses database structure to optimize keyword extraction
- **Multi-Strategy Fusion**: Combines semantic and keyword approaches for comprehensive coverage

#### 2. Advanced Retrieval Techniques

- **RAG Fusion**: Generates multiple subqueries for better recall
- **Knowledge Graph Integration**: Leverages structured entity relationships
- **Reciprocal Rank Fusion**: Optimally combines diverse search results

#### 3. Quality Optimization

- **Cross-Encoder Reranking**: Improves relevance through advanced scoring
- **Elbow Method Selection**: Dynamically determines optimal result count
- **Content Pruning**: Focuses on most relevant information

#### 4. Robust Answer Generation

- **Structured Prompting**: Uses engineered system prompts for consistent output
- **Citation Enforcement**: Ensures traceability and source attribution
- **Hallucination Prevention**: Strictly bounds responses to retrieved context

### Performance Characteristics

**Speed**: Optimized for fast retrieval with parallel processing
**Accuracy**: Multi-strategy approach ensures high recall and precision
**Scalability**: Efficient Elasticsearch integration handles large document collections
**Reliability**: Robust error handling and fallback mechanisms

### Integration Points

**Elasticsearch Backend**:

- Dense vector fields for semantic search
- Nested objects for knowledge graph data
- Optimized mappings for fast retrieval

**OpenAI Integration**:

- Embedding generation (text-embedding-3-large)
- LLM synthesis (GPT-4o-mini/GPT-4o)
- Query classification and optimization

**Knowledge Graph Support**:

- Entity and relationship extraction during ingestion
- Embedded entity descriptions for semantic search
- Graph traversal for complex queries

This Normal RAG architecture provides a solid foundation for accurate, fast document retrieval while maintaining the flexibility to handle diverse query types and document structures.

---

A. Ingestion:

RowBlaze is designed to efficiently ingest, process, and index both structured and unstructured data for Retrieval-Augmented Generation (RAG) applications. Below is a detailed breakdown of the ingestion pipeline, highlighting each stage, the tools involved, and the rationale behind the approach.

1. File Intake & Preprocessing

- Supported Formats:
  - Unstructured: PDF
  - Structured: PDF, CSV, XLSX
- File Handling:
  - Files are read as bytes, ensuring compatibility with various parsers and libraries.
  - Each file is assigned a unique document ID using a SHA256 hash of its content for traceability and deduplication.

---

2. Parser Selection

- Automatic Detection:
  - The pipeline detects the file type and selects the appropriate parser:
    - PDF: Uses pdfplumber and PyMuPDF for text, tables, and image extraction and description generation using openai.
    - OCR PDFs: If flagged or text extraction fails, uses pdf2image and pytesseract for OCR.
    - CSV/XLSX: Used custom parsers to extract rows and headers, preserving structure.

---

3. Content Extraction

- Text Extraction:
  - Extracts all readable text from each page (PDF) or row (CSV/XLSX).
- Table Extraction:
  - Converts tables into Markdown for consistent downstream processing.
- Image Extraction & Description:
  - Extracts images from PDFs and uses OpenAI Vision models to generate detailed descriptions, enriching the document context.

---

4. Chunking & Semantic Segmentation

- Chunking Strategy:
  - Uses langchain’s RecursiveCharacterTextSplitter to break content into manageable, context-aware chunks.
  - Chunk size and overlap are dynamically adjusted based on file type (e.g., larger for spreadsheets, smaller for dense text).
- Semantic Chunking for Structured Data:
  - For CSV/XLSX, chunks are created to preserve row context and header mapping, ensuring semantic integrity.

---

5. Document Summarization

- LLM-Powered Summaries:
  - The entire document is summarized using OpenAI’s language models, providing a high-level overview for each file.
  - If a user-provided summary is available, it is used as a fallback.

---

6. Chunk Enrichment & Knowledge Graph Extraction

- Chunk Enrichment:
  - Each chunk is optionally enriched using LLMs, leveraging context from neighboring chunks and the document summary.
- Knowledge Graph Extraction:
  - Entities and relationships are extracted from each chunk using prompt-engineered LLM calls, outputting a structured XML which is parsed into graph data.
  - Entity descriptions are embedded for semantic search.

---

7. Embedding Generation

- OpenAI Embeddings:
  - Each enriched chunk and entity description is embedded using OpenAI’s embedding models (configurable model/dimensions). Embeddings are used for vector search and semantic retrieval.

---

8. Indexing in Elasticsearch

- Schema Enforcement:
  - Ensures the Elasticsearch index exists and matches the expected schema, updating mappings if necessary.
- Bulk Ingestion:
  - All processed chunks (with text, embeddings, metadata, entities, and relationships) are ingested in bulk for efficiency. Detailed error handling and logging for traceability.

---

9. Cleanup & Resource Management

- Resource Handling:
  - All file and client resources are properly closed after processing.
  - Temporary files are cleaned up to avoid clutter.

---

libraries used -

1. Parsing & Extraction:

   - pdfplumber, PyMuPDF (fitz), pdf2image, pytesseract, openpyxl, csv

2. LLM Integration:

   - openai (Async API), prompt templates (YAML)

3. Chunking:

   - langchain’s RecursiveCharacterTextSplitter, tiktoken

4. Vector Search:

   - elasticsearch (Async client), dense vector fields

5. Orchestration:
   - asyncio for concurrency, robust error handling

---

B. Retrieval:

RowBlaze’s retrieval engine is designed to deliver accurate, explainable, and context-rich answers from both structured and unstructured data. The retrieval pipeline leverages advanced search, reranking, and LLM synthesis to maximize answer quality and traceability.

1. User Query Intake & Classification

- Query Reception:
  - Accepts natural language questions from users.
- Intent Classification:
  - Uses an LLM to classify the query (e.g., factual lookup, summary, comparison, complex analysis).
  - This classification guides the retrieval and ranking strategy.

---

2. Schema-Aware Keyword Extraction

- Schema Sampling:
  - Fetches representative data samples from the index to understand available fields and vocabulary.
- LLM-Guided Keyword Extraction:
  - Uses an LLM to extract the most relevant keyword or value from the user query, considering the schema and data samples.
  - Ensures precise filtering for factual lookups.

---

3. Subquery Generation (for Complex Queries)

- RAG Fusion Prompting:
  - For complex or broad queries, generates multiple focused subqueries using an LLM.
  - Each subquery targets a specific aspect of the original question, improving recall and coverage.

---

4. Semantic and Keyword Search

- Embedding Generation:
  - Converts queries and subqueries into dense vector embeddings using OpenAI models.
- Semantic Search:
  - Performs vector-based search over chunk embeddings and knowledge graph entity embeddings in Elasticsearch.
- Keyword Search:
  - Executes precise phrase or fuzzy keyword searches for exact matches and high-precision retrieval.

---

5. Knowledge Graph Retrieval

- Entity & Relationship Search:
  -Retrieves top-matching entities and relationships from the knowledge graph using both vector and keyword search.
  - Filters and ranks entities based on semantic similarity to the query.

---

6. Result Fusion & Ranking

- Reciprocal Rank Fusion (RRF):
  - Combines results from multiple search strategies (semantic, keyword, KG) using RRF to maximize diversity and relevance.
- Reranking (Optional as of now):
  -For deep research, applies a local cross-encoder reranker (e.g., BAAI/bge-reranker-base) to further refine result order based on query relevance.
- Pruning (Optional as of now):
  - Uses advanced models (e.g., Provence) to prune irrelevant content from retrieved chunks, focusing on the most pertinent information.

---

7. Context Formatting for LLM Synthesis

- Context Assembly:
  - Formats retrieved chunks and knowledge graph data into a structured, markdown-based context.
  - Includes source file names, page numbers, and chunk indices for traceability.
- Citation Management:
  - Tracks and highlights all sources used, enforcing citation at the paragraph level in the final answer.

---

8. Final Answer Generation

- LLM Synthesis:
  - Sends the assembled context and user query to an LLM (e.g., GPT-4o) with a detailed system prompt.
  - The LLM synthesizes a comprehensive, well-structured answer, strictly grounded in the retrieved context.
  - Citations are embedded at the end of each paragraph and summarized at the end.

---

9. Resource Cleanup

- Connection Management:
  - Ensures all Elasticsearch and OpenAI client connections are properly closed after each request.

---

- Libraries Used

1. Search & Vector Retrieval:

- elasticsearch (Async client), dense vector fields

2. LLM Integration:

- openai (Async API), prompt templates (YAML)

3. Reranking & Pruning:

- sentence-transformers (CrossEncoder), Provence model (optional)

4. Orchestration & Utilities:

- asyncio for concurrency, numpy for vector math, robust error handling

5. Context Formatting:

- Markdown formatting, citation management

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
