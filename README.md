# 🧠 Enterprise-Grade RAG Ingestion Pipeline with Milvus

This project provides a fully modular, production-ready **Retrieval-Augmented Generation (RAG)** ingestion pipeline that ingests domain-specific documents, embeds them using transformer models, and stores them in **Milvus** for fast and semantic similarity search.

It supports **batch ingestion**, **deduplication**, **vector normalization**, and **cross-encoder reranking**, all wrapped in a powerful CLI tool.

---

## 📦 Project Structure

```bash
.
├── rag_cli_tool.py           # Main CLI tool using Typer
├── run_ingestion_pipeline.py # Orchestration script for the full ingestion pipeline
├── docs.json                 # Example input document file
├── vectors.json              # Output embeddings file (auto-generated)
└── README.md                 # This documentation
```

---

## 🚀 What It Does

- Ingests JSON-formatted documents with `{"text": "..."}`
- Embeds text into dense vectors using `SentenceTransformer`
- Normalizes vectors for cosine similarity search
- Inserts into Milvus vector DB with metadata
- Supports semantic search with optional reranking

---

## 📌 Core Features

### 🧩 Data Processing
- ✅ JSON input ingestion (`docs.json`)
- ✅ SHA-256 deduplication with `hash_id`
- ✅ Document embedding and vector normalization

### 💾 Milvus Integration
- ✅ CLI-based Milvus connection
- ✅ Collection schema creation
- ✅ Batch inserts and vector indexing (HNSW)

### 🔎 Semantic Search
- ✅ Vector-based ANN search via Milvus
- ✅ Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`
- ✅ Top-k retrieval with confidence scores

### 🧰 CLI Management
- Modular commands:
  - `connect` to Milvus
  - `init-collection` schema setup
  - `embed` for embeddings
  - `insert` and `batch-insert`
  - `search` with reranking
  - `create-index` for post-insertion optimization

### ⚙️ Orchestration
- One-step execution via:
```bash
python run_ingestion_pipeline.py
```

---

## 📋 Prerequisites

### 🔧 System
- Python 3.8+
- Docker (optional for Milvus)
- 8GB+ RAM recommended

### 📦 Python Libraries
Install all required packages:
```bash
pip install typer sentence-transformers pymilvus cross-encoder tqdm
```

### 🧠 Milvus Server
- Self-hosted (via Docker Compose) or
- Zilliz Cloud instance

Example local Milvus:
```bash
docker compose up -d
```

---

## 🧪 How to Run

### 1. Run full pipeline
```bash
python run_ingestion_pipeline.py
```

### 2. Use CLI tool manually
```bash
python rag_cli_tool.py connect --host localhost --port 19530
python rag_cli_tool.py init-collection --collection rag_docs
python rag_cli_tool.py embed --input-file docs.json --output-file vectors.json
python rag_cli_tool.py insert --collection rag_docs --input-file vectors.json
```

### 3. Search from CLI
```bash
python rag_cli_tool.py search --collection rag_docs --query "How do transformers work?"
```

---

## 🧱 Extensibility

- ✅ Add custom metadata fields: `timestamp`, `document_id`, etc.
- ✅ Plug into AWS Lambda or Batch
- ✅ Dockerize for CI/CD pipelines
- ✅ Wrap as REST API using FastAPI

---

## 👥 Authors & Contributions

Built by AI architects for production-grade GenAI search and retrieval systems. Contributions welcome.

---

## 🛡 License

Apache 2.0 - Open-source, production-safe.

