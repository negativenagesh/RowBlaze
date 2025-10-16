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
OPENAI_MODEL=gpt-4o-mini-2024-07-18
OPENAI_SUMMARY_MODEL=gpt-4.1-nano-2025-04-14
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072
OPENAI_BASE_DIMENSION=3072

CHUNK_SIZE_TOKENS = 1024 
CHUNK_OVERLAP_TOKENS = 512 
FORWARD_CHUNKS = 3
BACKWARD_CHUNKS = 3
CHARS_PER_TOKEN_ESTIMATE = 4 
SUMMARY_MAX_TOKENS = 1024

SEMANTIC_NEIGHBORS=10
SEMANTIC_SIMILARITY_THRESHOLD=0.7

RAG_UPLOAD_ELASTIC_URL=xxxxxxx
ELASTICSEARCH_API_KEY=xxxxxxxx

MISTRAL_API_KEY=xxxx

JWT_SECRET_KEY=xxxxxxx

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

## Document Ingestion Pipeline

RowBlaze implements a sophisticated **multi-stage ingestion pipeline** that transforms raw documents into searchable, semantically-rich knowledge bases. The system handles both structured and unstructured data through specialized parsers, advanced chunking strategies, and comprehensive knowledge extraction.

### Ingestion Architecture Overview

The ingestion pipeline (`src/core/ingestion/rag_ingestion.py`) orchestrates a complex workflow that:

1. **Intelligently parses** diverse document formats with specialized processors
2. **Extracts and enriches** content using LLMs with contextual awareness
3. **Generates knowledge graphs** from unstructured text with entity/relationship extraction
4. **Creates semantic embeddings** for vector search using OpenAI models
5. **Applies document-level deduplication** for consistency across chunks
6. **Indexes optimally** in Elasticsearch with dynamic schema management

### How the Ingestion Process Works

#### 1. File Upload & Authentication (`api/routes/ingestion.py`)

```python
@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_name: str = Form(...),
    # ... authentication and parameters
):
```

**Process Flow**:

- **Authentication**: Validates user credentials via JWT tokens
- **File Validation**: Checks file type against supported formats
- **Content Hashing**: Generates SHA256 hash for deduplication
- **Background Processing**: Queues document for asynchronous processing
- **Immediate Response**: Returns document ID while processing continues

#### 2. Document ID Generation & Deduplication

```python
def _generate_doc_id_from_content(content_bytes: bytes) -> str:
    """Generates a SHA256 hash for the given byte content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()
```

**Key Features**:

- **Content-based hashing** prevents duplicate document ingestion
- **Consistent identification** across multiple uploads of same file
- **Traceability** throughout the entire processing pipeline

### Supported Document Formats

RowBlaze supports a comprehensive range of document types:

**Text Documents**:

- **PDF**: Standard and OCR-based (scanned documents)
- **DOC/DOCX**: Microsoft Word documents with full formatting preservation
- **ODT**: OpenDocument text files
- **TXT**: Plain text files

**Structured Data**:

- **CSV**: Comma-separated values with semantic chunking
- **XLSX**: Excel spreadsheets with intelligent table processing

**Images**:

- **JPG, PNG, GIF, BMP, WebP, HEIC, TIFF**: Vision-based content extraction

### Stage 1: File Intake & Preprocessing

#### Document ID Generation

```python
def _generate_doc_id_from_content(content_bytes: bytes) -> str:
    """Generates a SHA256 hash for the given byte content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()
```

**Key Features**:

- **Content-based hashing** ensures identical documents get the same ID
- **Deduplication prevention** at the document level
- **Traceability** throughout the entire pipeline

#### File Type Detection & Routing

The system automatically detects file types and routes to specialized processors:

```python
# Automatic processor selection based on file extension
if file_extension == ".pdf":
    doc_iterator = processor.process_pdf(...)
elif file_extension in [".jpg", ".jpeg", ".png", ...]:
    doc_iterator = processor.process_image(...)
elif file_extension == ".xlsx":
    doc_iterator = processor.process_xlsx_semantic_chunking(...)
```

### Stage 2: Specialized Content Extraction

#### PDF Processing (`PDFParser` class)

**Multi-Modal Extraction**:

```python
async def ingest(self, data: bytes) -> AsyncGenerator[Tuple[str, int], None]:
    # 1. Extract text from each page
    page_text = p_page.extract_text()

    # 2. Extract tables as separate blocks
    tables = p_page.extract_tables()
    table_markdown = self._convert_table_to_markdown(table)

    # 3. Extract and describe images
    image_list = fitz_page.get_images(full=True)
    description = await self._get_image_description(image_bytes)
```

**Advanced Features**:

- **Dual-library approach**: Uses both `pdfplumber` and `PyMuPDF` for comprehensive extraction
- **Table-to-Markdown conversion**: Preserves table structure for downstream processing
- **Vision-based image description**: Uses OpenAI Vision API to generate detailed image descriptions
- **OCR fallback**: Automatic OCR processing for scanned documents using Tesseract or Mistral OCR

#### Structured Data Processing

**CSV Semantic Chunking**:

```python
async def _create_semantic_csv_chunks(self, header_row: str, data_rows: List[str], file_name: str):
    # Creates context-aware chunks preserving header-row relationships
    chunk_text = f"CSV Structure:\n{header_row}\n\nData:\n" + "\n".join(batch_rows)
    context = f"This is part {part_num} of CSV file '{file_name}' containing rows {start}-{end}."
```

**XLSX Advanced Processing**:

```python
async def _create_semantic_xlsx_chunks(self, data_rows: List[List[str]], file_name: str):
    # Intelligent row batching with header preservation
    # Token-aware chunking to prevent context overflow
    # Header-value mapping for semantic integrity
```

**Key Innovations**:

- **Header context preservation**: Each chunk includes column headers for context
- **Token-aware batching**: Dynamically calculates optimal rows per chunk
- **Semantic integrity**: Maintains meaningful data relationships across chunks

#### Image Processing (`ImageParser` class)

**Vision-Based Content Extraction**:

```python
async def ingest(self, data: bytes, filename: str = None) -> AsyncGenerator[str, None]:
    # Uses OpenAI Vision API to generate detailed descriptions
    # Handles multiple image formats
    # Provides fallback descriptions for processing failures
```

### Stage 3: Intelligent Chunking Strategies

#### Dynamic Chunking Configuration

The system adapts chunking parameters based on document type:

```python
if file_extension in [".docx", ".doc", ".odt"]:
    chunk_size = 2048
    chunk_overlap = 1024
else:
    chunk_size = CHUNK_SIZE_TOKENS  # 20000
    chunk_overlap = CHUNK_OVERLAP_TOKENS  # 0
```

**Chunking Features**:

- **Token-based splitting**: Uses `tiktoken` for accurate token counting
- **Recursive character splitting**: Preserves semantic boundaries
- **Context-aware separators**: Prioritizes natural break points (`\n|`, `\n`, `|`, `. `)
- **File-type optimization**: Different strategies for different document types

#### Page-Aware Chunking

For documents with page structure:

```python
async def _generate_all_raw_chunks_from_doc(self, doc_text: str, file_name: str, doc_id: str, page_breaks: List[int] = None):
    # Assigns accurate page numbers to chunks
    # Maintains page context throughout processing
    # Handles documents with and without explicit page breaks
```

### Stage 4: LLM-Powered Content Enhancement

#### Document Summarization

**Comprehensive Document Analysis**:

```python
async def _generate_document_summary(self, full_document_text: str) -> str:
    # Uses GPT-4o-mini to create high-level document overviews
    # Provides context for chunk enrichment and knowledge extraction
    # Fallback to user-provided summaries when available
```

#### Chunk Enrichment

**Context-Aware Enhancement**:

```python
async def _enrich_chunk_content(self, chunk_text: str, document_summary: str,
                               preceding_chunks_texts: List[str],
                               succeeding_chunks_texts: List[str]) -> str:
    # Enriches chunks using surrounding context
    # Leverages document summary for global context
    # Maintains chunk size constraints while adding value
```

**Enrichment Strategy**:

- **Contextual awareness**: Uses 3 preceding and 3 succeeding chunks for context
- **Document-level context**: Incorporates overall document summary
- **Selective application**: Skips enrichment for tabular data and OCR content to preserve accuracy

### Stage 5: Knowledge Graph Extraction

#### Entity and Relationship Extraction

**Structured Knowledge Mining**:

```python
async def _extract_knowledge_graph(self, chunk_text: str, document_summary: str) -> Tuple[List[Dict], List[Dict]]:
    # Uses prompt-engineered LLM calls to extract entities and relationships
    # Outputs structured XML that's parsed into graph data
    # Applies sophisticated deduplication algorithms
```

**XML-Based Extraction Pipeline**:

```python
def _parse_graph_xml(self, xml_string: str) -> Tuple[List[Dict], List[Dict]]:
    # Robust XML parsing with fallback regex extraction
    # Handles malformed LLM outputs gracefully
    # Extracts entities with types, descriptions, and relationships
```

#### Hierarchical Structure Detection

**Advanced Hierarchy Extraction**:

```python
async def _extract_hierarchies(self, chunk_text: str, document_summary: str) -> List[Dict]:
    # Identifies organizational structures, taxonomies, and nested relationships
    # Creates multi-level hierarchy representations
    # Links hierarchical elements through structured relationships
```

#### Comprehensive Deduplication System

**Multi-Level Deduplication**:

1. **Chunk-Level Deduplication**:

```python
def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
    # Normalizes entity names and types for comparison
    # Merges descriptions from duplicate entities
    # Preserves embeddings from first occurrence
```

2. **Document-Level Deduplication**:

```python
def _apply_document_level_deduplication(self, processed_chunks: List[Dict], file_name: str):
    # Ensures consistency across entire document
    # Creates unified entity and relationship maps
    # Applies consistent representations to all chunks
```

**Deduplication Features**:

- **Fuzzy matching**: Handles variations in entity names and types
- **Description merging**: Combines information from duplicate entities
- **Relationship consolidation**: Merges similar relationships with weight preservation
- **Cross-chunk consistency**: Ensures same entities appear identically throughout document

### Stage 6: Embedding Generation

#### Multi-Modal Embedding Strategy

**Comprehensive Vector Generation**:

```python
async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    # Generates embeddings for chunk text and entity descriptions
    # Uses OpenAI text-embedding-3-large (3072 dimensions)
    # Handles batch processing for efficiency
```

**Embedding Applications**:

- **Chunk embeddings**: For semantic similarity search
- **Entity description embeddings**: For knowledge graph semantic search
- **Batch processing**: Efficient API usage with proper error handling
- **Fallback handling**: Graceful degradation when embedding generation fails

### Stage 7: Elasticsearch Indexing

#### Dynamic Schema Management

**Intelligent Index Creation**:

```python
async def ensure_es_index_exists(client: Any, index_name: str, mappings_body: Dict):
    # Creates indices with optimized mappings
    # Updates existing indices with new fields
    # Handles embedding dimension configuration
```

**Advanced Mapping Structure**:

```python
CHUNKED_PDF_MAPPINGS = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 3072, "similarity": "cosine"},
            "metadata": {
                "properties": {
                    "entities": {"type": "nested", "properties": {...}},
                    "relationships": {"type": "nested", "properties": {...}},
                    "hierarchies": {"type": "nested", "properties": {...}}
                }
            }
        }
    }
}
```

#### Bulk Ingestion Pipeline

**Efficient Data Loading**:

```python
# Bulk ingestion with comprehensive error handling
successes, response = await async_bulk(es_client, actions_for_es, raise_on_error=False)
```

**Ingestion Features**:

- **Bulk operations**: Efficient batch processing for large documents
- **Error resilience**: Detailed error reporting and partial success handling
- **Resource management**: Proper cleanup of connections and temporary files
- **Progress tracking**: Comprehensive logging throughout the process

### Stage 8: Quality Assurance & Optimization

#### Processing Optimization

**File-Type Specific Optimizations**:

- **PDF batching**: Large PDFs processed in 100-page batches to prevent memory issues
- **Concurrent processing**: Parallel chunk processing for improved performance
- **Resource management**: Automatic cleanup of temporary files and connections
- **Error recovery**: Graceful handling of processing failures with detailed logging

#### Content Quality Controls

**Processing Safeguards**:

- **OCR repetition cleaning**: Removes common OCR artifacts and repeated patterns
- **Content validation**: Ensures chunks meet minimum quality thresholds
- **Embedding validation**: Verifies successful embedding generation before indexing
- **Metadata integrity**: Maintains consistent metadata structure across all chunks

### Integration Points

#### API Integration (`api/routes/ingestion.py`)

**RESTful Ingestion Endpoint**:

```python
@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_name: str = Form(...),
    # ... additional parameters
):
    # Handles file uploads with authentication
    # Processes documents in background tasks
    # Returns immediate response with document ID
```

#### Streamlit Integration (`app/app.py`)

**User-Friendly Upload Interface**:

- **Drag-and-drop file upload** with format validation
- **Real-time processing status** with progress indicators
- **Document options configuration** (OCR, structured PDF, descriptions)
- **Advanced settings** for power users (chunk size, model selection)

### Performance Characteristics

**Scalability Features**:

- **Asynchronous processing**: Non-blocking I/O for high throughput
- **Memory efficiency**: Streaming processing for large documents
- **Batch optimization**: Intelligent batching for API efficiency
- **Resource pooling**: Efficient client connection management

**Quality Metrics**:

- **High accuracy**: Multi-modal extraction ensures comprehensive content capture
- **Consistency**: Document-level deduplication maintains data integrity
- **Traceability**: Complete audit trail from source to indexed content
- **Robustness**: Comprehensive error handling and recovery mechanisms

This ingestion pipeline represents a state-of-the-art approach to document processing, combining traditional NLP techniques with modern LLM capabilities to create rich, searchable knowledge bases from diverse document types.

---

## Document Ingestion Pipeline

RowBlaze implements a sophisticated **multi-stage ingestion pipeline** that transforms raw documents into searchable, semantically-rich knowledge bases. The system handles both structured and unstructured data through specialized parsers, advanced chunking strategies, and comprehensive knowledge extraction.

### Ingestion Architecture Overview

The ingestion pipeline (`src/core/ingestion/rag_ingestion.py`) orchestrates a complex workflow that:

1. **Intelligently parses** diverse document formats with specialized processors
2. **Extracts and enriches** content using LLMs with contextual awareness
3. **Generates knowledge graphs** from unstructured text with entity/relationship extraction
4. **Creates semantic embeddings** for vector search using OpenAI models
5. **Applies document-level deduplication** for consistency across chunks
6. **Indexes optimally** in Elasticsearch with dynamic schema management

### How the Ingestion Process Works

#### 1. File Upload & Authentication (`api/routes/ingestion.py`)

```python
@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_name: str = Form(...),
    # ... authentication and parameters
):
```

**Process Flow**:

- **Authentication**: Validates user credentials via JWT tokens
- **File Validation**: Checks file type against supported formats
- **Content Hashing**: Generates SHA256 hash for deduplication
- **Background Processing**: Queues document for asynchronous processing
- **Immediate Response**: Returns document ID while processing continues

#### 2. Document ID Generation & Deduplication

```python
def _generate_doc_id_from_content(content_bytes: bytes) -> str:
    """Generates a SHA256 hash for the given byte content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()
```

**Key Features**:

- **Content-based hashing** prevents duplicate document ingestion
- **Consistent identification** across multiple uploads of same file
- **Traceability** throughout the entire processing pipeline

### Supported Document Formats

RowBlaze supports a comprehensive range of document types:

**Text Documents**:

- **PDF**: Standard and OCR-based (scanned documents)
- **DOC/DOCX**: Microsoft Word documents with full formatting preservation
- **ODT**: OpenDocument text files
- **TXT**: Plain text files

**Structured Data**:

- **CSV**: Comma-separated values with semantic chunking
- **XLSX**: Excel spreadsheets with intelligent table processing

\*_Images_

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

---

## API Architecture & Endpoints

RowBlaze provides a comprehensive REST API built with FastAPI that handles authentication, document ingestion, retrieval, and chat management. The API is designed with security, scalability, and ease of use in mind.

### API Overview

The API is structured around four main modules:

- **Authentication API** (`/api/auth/*`): User management and JWT-based authentication
- **Ingestion API** (`/api/ingest`): Document upload and processing
- **Retrieval API** (`/api/query-*`): RAG-based question answering
- **Chat API** (`/api/chat/*`): Conversation history management

### Base Configuration

**API Base URL**: `http://localhost:8000/api` (development) or your deployed domain
**API Version**: v1
**Authentication**: JWT Bearer tokens
**Content Types**: JSON, multipart/form-data (for file uploads)

---

## Authentication API (`/api/auth/*`)

The authentication system provides secure user registration, login, and session management using JWT tokens.

### Endpoints

#### 1. User Registration

```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**Features:**

- Email format validation with regex patterns
- Password strength requirements (8+ chars, letters + numbers)
- Automatic user ID generation
- BCrypt password hashing
- Duplicate email prevention

#### 2. User Login

```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:** Same as registration

**Features:**

- Email/password authentication
- JWT token generation with configurable expiration
- Last login timestamp tracking
- Account status validation (disabled accounts rejected)

#### 3. OAuth2 Token Endpoint (Alternative)

```http
POST /api/auth/token
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=securepassword123
```

**OAuth2-compatible endpoint for standard authentication flows**

#### 4. Get Current User Info

```http
GET /api/auth/me
Authorization: Bearer <jwt_token>
```

**Response:**

```json
{
  "id": "user_123",
  "email": "user@example.com",
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-01T12:00:00Z",
  "email_verified": false
}
```

#### 5. Logout

```http
POST /api/auth/logout
Authorization: Bearer <jwt_token>
```

**Note:** Logout is client-side (token disposal). Server-side token blacklisting can be implemented for enhanced security.

### Security Features

- **JWT Tokens**: HS256 algorithm with configurable secret
- **Password Hashing**: BCrypt with salt generation
- **Input Validation**: Pydantic models with custom validators
- **Rate Limiting**: Can be implemented with middleware
- **CORS**: Configurable cross-origin resource sharing

---

## Ingestion API (`/api/ingest`)

The ingestion API handles document upload, processing, and indexing into Elasticsearch with comprehensive content extraction and knowledge graph generation.

### Document Upload Endpoint

```http
POST /api/ingest
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

file: <binary_file_data>
index_name: "user_documents"
description: "Optional document description"
is_ocr_pdf: false
is_structured_pdf: true
model: "gpt-4o-mini"
max_tokens: 16384
```

**Response:**

```json
{
  "success": true,
  "message": "Successfully processed document.pdf",
  "document_id": "sha256_hash_of_content"
}
```

### Supported File Types

**Text Documents:**

- PDF (standard and OCR-scanned)
- DOC/DOCX (Microsoft Word)
- ODT (OpenDocument Text)
- TXT (Plain text)

**Structured Data:**

- CSV (Comma-separated values)
- XLSX (Excel spreadsheets)

**Images:**

- JPG, JPEG, PNG, GIF, BMP, WebP, HEIC, TIFF

### Processing Pipeline

1. **Authentication & Validation**

   - JWT token verification
   - File type validation
   - Content hash generation for deduplication

2. **Background Processing**

   - Asynchronous document processing
   - Specialized parser selection based on file type
   - Multi-modal content extraction (text, tables, images)

3. **Content Enhancement**

   - LLM-powered document summarization
   - Chunk enrichment with contextual information
   - Knowledge graph extraction (entities, relationships, hierarchies)

4. **Indexing**
   - Embedding generation using OpenAI models
   - Elasticsearch bulk ingestion
   - Schema management and optimization

### File Management Endpoints

#### List Indexed Files

```http
GET /api/files/{index_name}
Authorization: Bearer <jwt_token>
```

**Response:**

```json
["document1.pdf", "spreadsheet.xlsx", "report.docx"]
```

#### Health Check

```http
GET /api/health
```

**Response:**

```json
{
  "status": "ok",
  "message": "API is running"
}
```

---

## Retrieval API (`/api/query-*`)

The retrieval API provides both Normal RAG and Agentic RAG capabilities for intelligent question answering over indexed documents.

### Normal RAG Endpoint

```http
POST /api/query-rag
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "question": "What are the key findings in the research report?",
  "index_name": "user_documents",
  "top_k_chunks": 10,
  "enable_references_citations": true,
  "deep_research": false,
  "auto_chunk_sizing": true,
  "model": "gpt-4o-mini",
  "max_tokens": 16384
}
```

**Response:**

```json
{
  "question": "What are the key findings in the research report?",
  "answer": "Based on the research report, the key findings include...\n\n**Source:** document.pdf (Page 5)",
  "context": [
    {
      "text": "Relevant chunk content...",
      "score": 0.95,
      "file_name": "document.pdf",
      "page_number": 5,
      "entities": [...],
      "relationships": [...]
    }
  ],
  "cited_files": ["document.pdf"],
  "metadata": {
    "query_type": "summary_extraction",
    "chunks_retrieved": 10,
    "processing_time": 2.3
  }
}
```

### Agentic RAG Endpoint

```http
POST /api/agent-query
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "question": "Compare the financial performance across different quarters",
  "index_name": "user_documents",
  "top_k_chunks": 20,
  "enable_references_citations": true,
  "deep_research": true,
  "model": "gpt-4o-mini",
  "max_tokens": 16384
}
```

**Response:**

```json
{
  "question": "Compare the financial performance across different quarters",
  "answer": "Comprehensive analysis with multi-step reasoning...",
  "context": [...],
  "cited_files": [...],
  "metadata": {
    "agent_mode": "pure_agentic_rag",
    "query_type": "comparison",
    "iterations_completed": 2,
    "tools_used": ["search_file_knowledge", "vector_search", "keyword_search"],
    "initial_retrieval_performed": false,
    "processing_time": 8.7
  }
}
```

### RAG Modes Comparison

| Feature        | Normal RAG       | Agentic RAG           |
| -------------- | ---------------- | --------------------- |
| **Speed**      | Fast (2-4s)      | Slower (5-15s)        |
| **Complexity** | Simple queries   | Complex analysis      |
| **Tool Usage** | Single retrieval | Multi-tool selection  |
| **Iterations** | Single pass      | Up to 2 iterations    |
| **Best For**   | Direct questions | Multi-faceted queries |

### Advanced Retrieval Features

#### Document Retrieval

```http
POST /api/query-documents
Authorization: Bearer <jwt_token>

{
  "question": "Find information about project timelines",
  "index_name": "user_documents",
  "top_k_chunks": 15
}
```

#### Final Answer Generation

```http
POST /api/generate-final-answer
Authorization: Bearer <jwt_token>

{
  "question": "Original question",
  "context": "Retrieved context with sources",
  "generate_final_answer": true,
  "enable_references_citations": true
}
```

### Data Exploration Endpoints

#### View Document Chunks

```http
GET /api/chunks/{index_name}?file_name=document.pdf
Authorization: Bearer <jwt_token>
```

#### View Knowledge Graph

```http
GET /api/knowledge-graph/{index_name}?file_name=document.pdf
Authorization: Bearer <jwt_token>
```

**Response:**

```json
{
  "success": true,
  "knowledge_graph": {
    "entities": [
      {
        "name": "Company ABC",
        "type": "Organization",
        "description": "Technology company...",
        "file_name": "document.pdf",
        "page_number": 3
      }
    ],
    "relationships": [
      {
        "source_entity": "Company ABC",
        "relation": "acquired",
        "target_entity": "Startup XYZ",
        "file_name": "document.pdf"
      }
    ],
    "hierarchies": [...]
  }
}
```

---

## Chat API (`/api/chat/*`)

The chat API manages conversation history and session persistence for seamless user interactions.

### Chat Session Management

#### Save Chat History

```http
POST /api/chat/save
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "session_id": "unique_session_id",
  "title": "Discussion about Q3 Results",
  "messages": [
    {
      "role": "user",
      "content": "What were the Q3 results?",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "The Q3 results showed...",
      "timestamp": "2024-01-01T12:00:05Z"
    }
  ]
}
```

#### Retrieve Chat History

```http
GET /api/chat/{session_id}
Authorization: Bearer <jwt_token>
```

**Response:**

```json
{
  "success": true,
  "messages": [
    {
      "role": "user",
      "content": "What were the Q3 results?",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "The Q3 results showed...",
      "timestamp": "2024-01-01T12:00:05Z"
    }
  ]
}
```

#### List All Chat Sessions

```http
GET /api/chat/list/sessions
Authorization: Bearer <jwt_token>
```

**Response:**

```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "session_123",
      "title": "Discussion about Q3 Results",
      "last_updated": "2024-01-01T12:00:00Z",
      "message_count": 6
    }
  ]
}
```

#### Delete Chat Session

```http
DELETE /api/chat/{session_id}
Authorization: Bearer <jwt_token>
```

### Chat Features

- **User Isolation**: Each user can only access their own chat sessions
- **Automatic Titles**: Generated from first user message if not provided
- **Elasticsearch Storage**: Scalable chat history persistence
- **Session Management**: Create, read, update, delete operations
- **Timestamp Tracking**: Automatic message and session timestamping

---

## CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

RowBlaze includes a comprehensive CI/CD pipeline that ensures code quality, security, and automated deployment.

### Pipeline Stages

#### 1. **Testing & Quality Assurance**

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - Python 3.11 setup with uv package manager
      - Dependency caching for faster builds
      - Code linting with flake8 (syntax errors, undefined names)
      - Code formatting with Black
      - Import sorting with isort
      - Type checking with mypy
      - Test execution with pytest and coverage reporting
      - Coverage upload to Codecov
```

**Quality Gates:**

- **Linting**: Catches Python syntax errors and undefined names
- **Formatting**: Ensures consistent code style with Black
- **Import Sorting**: Maintains organized imports with isort
- **Type Checking**: Static type analysis with mypy
- **Test Coverage**: Comprehensive test suite with coverage reporting

#### 2. **Docker Build & Security Scanning**

```yaml
build-scan-and-push:
  needs: test
  if: github.ref == 'refs/heads/main'
  steps:
    - Multi-stage Docker builds for API, App, and Nginx
    - GitHub Container Registry (ghcr.io) integration
    - Trivy security scanning for vulnerabilities
    - SARIF upload for security analysis
    - Automated image pushing on successful builds
```

**Security Features:**

- **Vulnerability Scanning**: Trivy scans for CRITICAL and HIGH severity issues
- **SARIF Integration**: Security findings uploaded to GitHub Security tab
- **Multi-stage Builds**: Optimized Docker images with minimal attack surface
- **Registry Security**: Secure image storage in GitHub Container Registry

#### 3. **Deployment Automation**

```yaml
deploy:
  needs: build-scan-and-push
  if: github.ref == 'refs/heads/main'
  steps:
    - Production deployment triggers
    - Environment-specific configurations
    - Health check validations
```

### Container Architecture

#### API Container (`Dockerfile.api`)

```dockerfile
# Multi-stage build for optimized production image
FROM python:3.11-slim AS builder
# Virtual environment creation and dependency installation
FROM python:3.11-slim AS final
# Production runtime with health checks
```

**Features:**

- **Multi-stage Build**: Separates build dependencies from runtime
- **Virtual Environment**: Isolated Python dependencies
- **Health Checks**: Automated container health monitoring
- **Security**: Non-root user execution and minimal base image

#### App Container (`Dockerfile.app`)

```dockerfile
# Streamlit application container
# Health check via Streamlit's built-in endpoint
# File upload directory creation with proper permissions
```

#### Nginx Container (`nginx/Dockerfile`)

```dockerfile
# Reverse proxy and load balancer
# SSL termination and static file serving
# Request routing between API and App containers
```

### Environment Configuration

**Development:**

```bash
# Local development with hot reloading
uv run src/core/ingestion/rag_ingestion.py
streamlit run app/app.py
```

**Production (Docker Compose):**

```bash
docker-compose up --build
# Orchestrates API, App, and Nginx containers
# Shared networking and volume management
# Health checks and restart policies
```

**Individual Containers:**

```bash
# API
docker build -f Dockerfile.api -t rowblaze-api .
docker run -p 8000:8000 --env-file .env rowblaze-api

# App
docker build -f Dockerfile.app -t rowblaze-app .
docker run -p 8501:8501 --env-file .env rowblaze-app

# Nginx
docker build -f nginx/Dockerfile -t rowblaze-nginx ./nginx
docker run -p 80:80 rowblaze-nginx
```

### Monitoring & Observability

**Health Endpoints:**

- **API**: `GET /api/health` - Service status and version info
- **App**: Streamlit's `/_stcore/health` - Frontend health check
- **Nginx**: Basic connectivity and routing validation

**Logging:**

- **Structured Logging**: JSON format with timestamps and levels
- **Request Tracing**: Correlation IDs for request tracking
- **Error Handling**: Comprehensive exception logging with stack traces

**Metrics:**

- **Container Health**: Docker health check status
- **Response Times**: API endpoint performance monitoring
- **Error Rates**: HTTP status code tracking
- **Resource Usage**: CPU, memory, and disk utilization

### Security Best Practices

**Code Security:**

- **Dependency Scanning**: Automated vulnerability detection
- **Secret Management**: Environment variable configuration
- **Input Validation**: Pydantic models with strict validation
- **Authentication**: JWT-based secure authentication

**Infrastructure Security:**

- **Container Scanning**: Trivy security analysis
- **Network Isolation**: Docker network segmentation
- **SSL/TLS**: HTTPS encryption for production deployments
- **Access Control**: Role-based permissions and user isolation

This comprehensive CI/CD pipeline ensures that RowBlaze maintains high code quality, security standards, and reliable deployments while supporting both development and production environments.

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
