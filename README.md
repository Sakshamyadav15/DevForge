# 🔍 Hybrid Search Engine

A powerful **hybrid search system** combining **vector similarity search** (FAISS) with **graph-based search** (Neo4j) to deliver intelligent, context-aware results.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?logo=neo4j)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

This project implements a **hybrid search engine** that combines two powerful search paradigms:

1. **Vector Search (Semantic)**: Uses embeddings to find semantically similar content
2. **Graph Search (Relational)**: Traverses relationships between entities to find connected information

The system **automatically decides** how to weight these two approaches based on your query, giving you the best of both worlds without manual tuning.

### Why Hybrid Search?

| Search Type | Best For | Limitation |
|-------------|----------|------------|
| **Vector Search** | Finding similar content, semantic matching | Misses relationships |
| **Graph Search** | Finding connected entities, traversing relationships | Misses semantic similarity |
| **Hybrid Search** | ✅ Both! Combines semantic + relational intelligence | None! |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│                   Vite + TypeScript + shadcn/ui                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Ingestion  │  │   Search    │  │     Hybrid Engine       │ │
│  │     API     │  │    API      │  │  (Adaptive Weights)     │ │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
│         │                │                     │               │
│         ▼                ▼                     ▼               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Service Layer                            ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ ││
│  │  │ Embedding │  │  Vector   │  │   Graph   │  │ Snapshot │ ││
│  │  │  Service  │  │   Store   │  │   Store   │  │ Manager  │ ││
│  │  │(MiniLM-L6)│  │  (FAISS)  │  │  (Neo4j)  │  │  (JSON)  │ ││
│  │  └───────────┘  └───────────┘  └───────────┘  └──────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    FAISS    │    │   Neo4j     │    │ snapshot.json│
    │Vector Index │    │Graph Database│   │   (Backup)   │
    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## ✨ Features

### 🔎 Search Capabilities

| Feature | Description |
|---------|-------------|
| **Vector Search** | Find semantically similar documents using embeddings |
| **Graph Search** | Traverse node relationships with configurable depth |
| **Hybrid Search** | Combines both with **adaptive weights** |
| **Auto-Intent Detection** | System automatically determines optimal search strategy |

### 📥 Data Ingestion

- **Single Document Ingestion**: Add one document at a time
- **Bulk Ingestion**: Import multiple documents efficiently
- **Automatic Embedding**: Text → 384-dimensional vector
- **Automatic Relationships**: Similar documents are linked automatically

### 📊 Dashboard

- Real-time statistics (nodes, edges, vectors)
- Topic distribution charts
- Edge type analysis
- System health monitoring

---

## 🛠️ Tech Stack

### Backend

| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance REST API framework |
| **Neo4j** | Graph database for storing nodes and relationships |
| **FAISS** | Facebook's vector similarity search library |
| **Sentence Transformers** | Text embeddings (all-MiniLM-L6-v2) |
| **Pydantic** | Data validation and serialization |

### Frontend

| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Fast build tool and dev server |
| **TanStack Query** | Data fetching and caching |
| **shadcn/ui** | Beautiful, accessible UI components |
| **Tailwind CSS** | Utility-first styling |
| **Recharts** | Data visualization |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (or Bun)
- **Neo4j Database** (local or cloud)

### 1. Clone the Repository

```bash
git clone https://github.com/Sakshamyadav15/DevForge.git
cd DevForge
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the `backend` folder:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Data Paths
SNAPSHOT_PATH=data/snapshot.json
FAISS_INDEX_PATH=data/vector_index
```

### 4. Start Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 5. Frontend Setup

```bash
cd frontend

# Install dependencies (using npm)
npm install

# Or using bun
bun install
```

### 6. Start Frontend

```bash
npm run dev
# or
bun dev
```

The UI will be available at `http://localhost:5173`

---

## 📡 API Reference

### Search Endpoints

#### Vector Search
```http
POST /search/vector
Content-Type: application/json

{
  "query": "machine learning algorithms",
  "top_k": 10
}
```

#### Graph Search
```http
POST /search/graph/search
Content-Type: application/json

{
  "query": "neural networks",
  "top_k": 10,
  "max_depth": 2
}
```

#### Hybrid Search (Recommended)
```http
POST /search/hybrid
Content-Type: application/json

{
  "query": "how is deep learning related to AI?",
  "top_k": 10
}
```

> **Note**: Hybrid search automatically detects query intent and adjusts weights:
> - Relationship queries → 40% vector, 60% graph
> - Content queries → 70% vector, 30% graph

### Ingestion Endpoints

#### Ingest Single Document
```http
POST /ingest/node
Content-Type: application/json

{
  "text": "Machine learning is a subset of artificial intelligence...",
  "metadata": {
    "title": "ML Introduction",
    "topic": "AI",
    "category": "concept"
  }
}
```

#### Bulk Ingest
```http
POST /ingest/bulk
Content-Type: application/json

{
  "nodes": [
    {"text": "Document 1...", "metadata": {"topic": "AI"}},
    {"text": "Document 2...", "metadata": {"topic": "ML"}}
  ]
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/nodes` | GET | List all nodes (paginated) |
| `/nodes/{id}` | GET | Get single node |
| `/nodes/{id}/neighbors` | GET | Get node neighbors |
| `/edges` | GET | List all edges (paginated) |
| `/stats` | GET | Get database statistics |
| `/health` | GET | Health check |

---

## 🧠 How It Works

### 1. Document Ingestion Flow

```
User submits document
        │
        ▼
┌───────────────────┐
│ Generate Embedding │ ──► 384-dimensional vector
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Create Neo4j Node │ ──► Store text, topic, metadata
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Add to FAISS Index│ ──► Enable similarity search
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Find Similar Nodes│ ──► Search existing documents
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Create Edges      │ ──► SIMILAR_TO, RELATED_TO
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Save to Snapshot  │ ──► Backup to JSON
└───────────────────┘
```

### 2. Hybrid Search Flow

```
User Query: "How is deep learning related to neural networks?"
                    │
                    ▼
         ┌─────────────────────┐
         │ Detect Query Intent │
         └──────────┬──────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Contains "related", "connect" │
    │ "link", "between", etc?       │
    └───────────────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    [YES: Relationship]   [NO: Content]
         │                     │
    Vector: 40%           Vector: 70%
    Graph:  60%           Graph:  30%
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │     Execute Both Searches     │
    │  ┌─────────┐   ┌───────────┐  │
    │  │  FAISS  │   │   Neo4j   │  │
    │  │ Vector  │   │   Graph   │  │
    │  └────┬────┘   └─────┬─────┘  │
    └───────┼──────────────┼────────┘
            │              │
            ▼              ▼
    ┌───────────────────────────────┐
    │   Normalize & Combine Scores  │
    │   final = (v*w_v) + (g*w_g)   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │      Return Top K Results     │
    └───────────────────────────────┘
```

### 3. Embedding Model

We use **all-MiniLM-L6-v2** from Sentence Transformers:

- **Dimensions**: 384
- **Max Sequence**: 256 tokens
- **Speed**: ~14,000 sentences/sec on GPU
- **Quality**: Excellent for semantic similarity

### 4. Relationship Types

| Type | Similarity | Description |
|------|------------|-------------|
| `SIMILAR_TO` | > 0.8 | Very closely related content |
| `RELATED_TO` | 0.6 - 0.8 | Moderately related content |
| `MENTIONS` | 0.4 - 0.6 | Loosely related content |

---

## 📁 Project Structure

```
DevForge/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry
│   │   ├── config.py            # Configuration settings
│   │   ├── api/
│   │   │   ├── ingest.py        # Ingestion endpoints
│   │   │   ├── search.py        # Search endpoints
│   │   │   ├── nodes.py         # Node CRUD endpoints
│   │   │   └── edges.py         # Edge CRUD endpoints
│   │   ├── models/
│   │   │   ├── graph.py         # Node/Edge Pydantic models
│   │   │   └── search.py        # Search request/response models
│   │   └── services/
│   │       ├── embeddings.py    # Embedding generation
│   │       ├── vector_store.py  # FAISS operations
│   │       ├── graph_store.py   # Neo4j operations
│   │       ├── hybrid_engine.py # Hybrid search logic
│   │       └── snapshot.py      # JSON persistence
│   ├── data/
│   │   ├── snapshot.json        # Data backup
│   │   └── vector_index.faiss   # FAISS index
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main app component
│   │   ├── main.tsx             # Entry point
│   │   ├── components/
│   │   │   ├── AppSidebar.tsx   # Navigation sidebar
│   │   │   └── ui/              # shadcn/ui components
│   │   ├── pages/
│   │   │   ├── Search.tsx       # Search interface
│   │   │   ├── Ingestion.tsx    # Data ingestion form
│   │   │   ├── Nodes.tsx        # Node browser
│   │   │   ├── Edges.tsx        # Edge browser
│   │   │   ├── Stats.tsx        # Statistics dashboard
│   │   │   └── Overview.tsx     # System overview
│   │   ├── lib/
│   │   │   ├── api.ts           # API client functions
│   │   │   └── utils.ts         # Utility functions
│   │   └── types/
│   │       └── index.ts         # TypeScript types
│   ├── package.json
│   └── vite.config.ts
│
└── README.md                    # You are here!
```

---

## 🔧 Configuration Options

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `SNAPSHOT_PATH` | `data/snapshot.json` | Backup file path |
| `FAISS_INDEX_PATH` | `data/vector_index` | FAISS index path |

### Frontend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL |

---

## 📈 Performance Tips

1. **Batch Ingestion**: Use `/ingest/bulk` for multiple documents
2. **Index Size**: FAISS performs best with < 1M vectors
3. **Graph Depth**: Keep traversal depth ≤ 3 for speed
4. **Caching**: TanStack Query caches API responses automatically

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Saksham Yadav** - [GitHub](https://github.com/Sakshamyadav15)

---

<p align="center">
  Made with ❤️ for the DevForge Hackathon
</p>
