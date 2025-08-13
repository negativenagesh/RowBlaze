<div align='center'>
  <img src="RowBlaze-logo/cover.png" alt="RowBlaze Logo" width="800"/>
  <br>
  <b>RAG for Structured and Unstructured data</b>
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

<div align='center'>
  
[Demo (Structured data) - Click to watch](https://drive.google.com/file/d/1Z3UCeQxKMCFTWQoLPUvOWWFKhzPoK3Wp/view?usp=sharing)
</div>

<div align='center'>

[Demo (Unstructured data) - Click to watch](https://drive.google.com/file/d/1jUIGVub3BVo1jIIUP7Ao7wDp5XeAk6Sy/view?usp=sharing)
</div>

<img width="1680" height="887" alt="Screenshot 2025-08-11 at 00 21 11" src="https://github.com/user-attachments/assets/c7169b4e-d63c-4927-856c-194807d3dbb0" />
<img width="1680" height="875" alt="Screenshot 2025-08-11 at 00 22 49" src="https://github.com/user-attachments/assets/af181565-0c78-4916-88ab-3a4ccaa91e59" />
<img width="1680" height="885" alt="image" src="https://github.com/user-attachments/assets/0ab284dc-9722-4147-9754-dca297f4614c" />
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
source .venv/bin/activate
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
OPENAI_SUMMARY_MODEL=gpt-4.1-nano
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
streamlit app/app.py
```
---
### 7. Notes

- Ensure your Elasticsearch instance is running and accessible.
- Your OpenAI API key must have access to the specified models.
- For OCR, install `pytesseract` and ensure Tesseract is in your system PATH.
- For local reranking, download the model specified by `RERANKER_MODEL_ID` if not using the default.

---

### 8. Demo

See the demo links and screenshots at the top of this README for usage examples.

---

A. Ingestion:

RowBlaze is designed to efficiently ingest, process, and index both structured and unstructured data for Retrieval-Augmented Generation (RAG) applications. Below is a detailed breakdown of the ingestion pipeline, highlighting each stage, the tools involved, and the rationale behind the approach.

1. File Intake & Preprocessing
* Supported Formats:
    - Unstructured: PDF
    - Structured: PDF, CSV, XLSX
* File Handling:
    - Files are read as bytes, ensuring compatibility with various parsers and libraries.
    - Each file is assigned a unique document ID using a SHA256 hash of its content for traceability and deduplication.
---
2. Parser Selection
* Automatic Detection:
    - The pipeline detects the file type and selects the appropriate parser:
        - PDF: Uses pdfplumber and PyMuPDF for text, tables, and image extraction and description generation using openai.
        - OCR PDFs: If flagged or text extraction fails, uses pdf2image and pytesseract for OCR.
        - CSV/XLSX: Used custom parsers to extract rows and headers, preserving structure.
---
3. Content Extraction
* Text Extraction:
    - Extracts all readable text from each page (PDF) or row (CSV/XLSX).
* Table Extraction:
    - Converts tables into Markdown for consistent downstream processing.
* Image Extraction & Description:
    - Extracts images from PDFs and uses OpenAI Vision models to generate detailed descriptions, enriching the document context.
---
4. Chunking & Semantic Segmentation
* Chunking Strategy:
    - Uses langchain’s RecursiveCharacterTextSplitter to break content into manageable, context-aware chunks.
    - Chunk size and overlap are dynamically adjusted based on file type (e.g., larger for spreadsheets, smaller for dense text).
* Semantic Chunking for Structured Data:
    - For CSV/XLSX, chunks are created to preserve row context and header mapping, ensuring semantic integrity.
---
5. Document Summarization
* LLM-Powered Summaries:
    - The entire document is summarized using OpenAI’s language models, providing a high-level overview for each file.
    - If a user-provided summary is available, it is used as a fallback.
---
6. Chunk Enrichment & Knowledge Graph Extraction
* Chunk Enrichment:
    - Each chunk is optionally enriched using LLMs, leveraging context from neighboring chunks and the document summary.
* Knowledge Graph Extraction:
    - Entities and relationships are extracted from each chunk using prompt-engineered LLM calls, outputting a structured XML which is parsed into graph data.
    - Entity descriptions are embedded for semantic search.
---
7. Embedding Generation
* OpenAI Embeddings:
    - Each enriched chunk and entity description is embedded using OpenAI’s embedding models (configurable model/dimensions). Embeddings are used for vector search and semantic retrieval.
---
8. Indexing in Elasticsearch
* Schema Enforcement:
    - Ensures the Elasticsearch index exists and matches the expected schema, updating mappings if necessary.
* Bulk Ingestion:
    - All processed chunks (with text, embeddings, metadata, entities, and relationships) are ingested in bulk for efficiency. Detailed error handling and logging for traceability.
---
9. Cleanup & Resource Management
* Resource Handling:
    - All file and client resources are properly closed after processing.
    - Temporary files are cleaned up to avoid clutter.
---
libraries used - 
1. Parsing & Extraction:
    * pdfplumber, PyMuPDF (fitz), pdf2image, pytesseract, openpyxl, csv

2. LLM Integration:
    * openai (Async API), prompt templates (YAML)

3. Chunking:
    * langchain’s RecursiveCharacterTextSplitter, tiktoken

4. Vector Search:
    * elasticsearch (Async client), dense vector fields

5. Orchestration:
    * asyncio for concurrency, robust error handling
---
B. Retrieval:

RowBlaze’s retrieval engine is designed to deliver accurate, explainable, and context-rich answers from both structured and unstructured data. The retrieval pipeline leverages advanced search, reranking, and LLM synthesis to maximize answer quality and traceability.

1. User Query Intake & Classification
* Query Reception:
    - Accepts natural language questions from users.
* Intent Classification:
    - Uses an LLM to classify the query (e.g., factual lookup, summary, comparison, complex analysis).
    - This classification guides the retrieval and ranking strategy.
---
2. Schema-Aware Keyword Extraction
* Schema Sampling:
    - Fetches representative data samples from the index to understand available fields and vocabulary.
* LLM-Guided Keyword Extraction:
    - Uses an LLM to extract the most relevant keyword or value from the user query, considering the schema and data samples.
    - Ensures precise filtering for factual lookups.
---
3. Subquery Generation (for Complex Queries)
* RAG Fusion Prompting:
    - For complex or broad queries, generates multiple focused subqueries using an LLM.
    - Each subquery targets a specific aspect of the original question, improving recall and coverage.
---
4. Semantic and Keyword Search
* Embedding Generation:
    - Converts queries and subqueries into dense vector embeddings using OpenAI models.
* Semantic Search:
    - Performs vector-based search over chunk embeddings and knowledge graph entity embeddings in Elasticsearch.
* Keyword Search:
    - Executes precise phrase or fuzzy keyword searches for exact matches and high-precision retrieval.
---
5. Knowledge Graph Retrieval
* Entity & Relationship Search:
    -Retrieves top-matching entities and relationships from the knowledge graph using both vector and keyword search.
    - Filters and ranks entities based on semantic similarity to the query.
---
6. Result Fusion & Ranking
* Reciprocal Rank Fusion (RRF):
    - Combines results from multiple search strategies (semantic, keyword, KG) using RRF to maximize diversity and relevance.
* Reranking (Optional as of now):
    -For deep research, applies a local cross-encoder reranker (e.g., BAAI/bge-reranker-base) to further refine result order based on query relevance.
* Pruning (Optional as of now):
    - Uses advanced models (e.g., Provence) to prune irrelevant content from retrieved chunks, focusing on the most pertinent information.
---
7. Context Formatting for LLM Synthesis
* Context Assembly:
    - Formats retrieved chunks and knowledge graph data into a structured, markdown-based context.
    - Includes source file names, page numbers, and chunk indices for traceability.
* Citation Management:
    - Tracks and highlights all sources used, enforcing citation at the paragraph level in the final answer.
---
8. Final Answer Generation
* LLM Synthesis:
    - Sends the assembled context and user query to an LLM (e.g., GPT-4o) with a detailed system prompt.
    - The LLM synthesizes a comprehensive, well-structured answer, strictly grounded in the retrieved context.
    - Citations are embedded at the end of each paragraph and summarized at the end.
---
9. Resource Cleanup
* Connection Management:
    - Ensures all Elasticsearch and OpenAI client connections are properly closed after each request.
---
* Libraries Used
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


