<div align='center'>
  <img src="RowBlaze-logo/cover.png" alt="RowBlaze Logo" width="800"/>
  <br>
  <b> Most accurate RAG/Agentic RAG for Structured and Unstructured data</b>
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

## 🎥 Demo Video

Watch the complete walkthrough of RowBlaze features including signup/login, document upload, and RAG/Agentic RAG functionality:

<video controls width="100%" src="Screen Recording 2025-10-16 at 01.48.39.mov" title="RowBlaze Demo - Signup, Upload, and RAG Features">
  Your browser does not support the video tag. You can <a href="Screen Recording 2025-10-16 at 01.48.39.mov">download the video</a> instead.
</video>

*Note: This video demonstrates the complete user journey from authentication to document processing and querying with both Normal RAG and Agentic RAG modes.*

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
@@ -48,208 +28,41 @@ uv pip install -r requirements.txt
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

### 8. Demo

See the demo links and screenshots at the top of this README for usage examples.

---

A. Ingestion:

RowBlaze is designed to efficiently ingest, process, and index both structured and unstructured data for Retrieval-Augmented Generation (RAG) applications. Below is a detailed breakdown of the ingestion pipeline, highlighting each stage, the tools involved, and the rationale behind the approach.
