# DevForge: Hybrid Vector and Graph Search Engine

## Overview
DevForge is a hybrid search engine that integrates vector similarity search with graph-based traversal to provide context-aware information retrieval. By combining semantic understanding from vector embeddings with relational insights from a graph database, the system delivers search results that are both semantically relevant and structurally connected.

## Architecture
The system employs a microservices-based architecture with a decoupled frontend and backend.

- **Frontend**: A React-based single-page application (SPA) utilizing TypeScript and Vite for performance, with a UI component library for consistent design.
- **Backend**: A high-performance REST API built with FastAPI, orchestrating interactions between the vector store, graph database, and embedding services.
- **Data Layer**:
  - **Vector Store (FAISS)**: Manages high-dimensional vector embeddings for semantic similarity search.
  - **Graph Database (Neo4j)**: Stores entities and relationships to enable graph traversal and structural analysis.
  - **Snapshot Manager**: Provides persistence and recovery capabilities through JSON-based snapshots.

## Features

### Search Capabilities
- **Vector Search**: Retrieves documents based on semantic similarity using dense vector embeddings.
- **Graph Search**: Traverses node relationships to discover connected entities within a configurable depth.
- **Hybrid Search**: Synthesizes results from both vector and graph searches using adaptive weighting algorithms to optimize relevance based on query intent.

### Data Ingestion
- **Automated Embedding**: Converts textual content into 384-dimensional vectors using the all-MiniLM-L6-v2 model.
- **Relationship Extraction**: Automatically identifies and creates relationships (e.g., SIMILAR_TO, RELATED_TO) between documents based on content similarity.
- **Bulk Processing**: Supports efficient ingestion of large datasets through batch processing endpoints.

### System Monitoring
- **Dashboard**: Provides real-time visualization of system metrics, including node counts, edge distributions, and vector index status.
- **Health Checks**: Monitors the operational status of all subsystems, including database connections and service availability.

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Graph Database**: Neo4j
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Validation**: Pydantic

### Frontend
- **Framework**: React 18
- **Language**: TypeScript
- **Build Tool**: Vite
- **State Management**: TanStack Query
- **UI Library**: shadcn/ui
- **Styling**: Tailwind CSS

## Installation and Setup

### Prerequisites
- Python 3.11 or higher
- Node.js 18 or higher
- Neo4j Database instance (local or remote)

### Backend Configuration
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Unix/MacOS
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in a `.env` file:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```
5. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Configuration
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## API Reference

### Search Operations
- `POST /search/vector`: Performs semantic search using vector embeddings.
- `POST /search/graph`: Executes graph traversal queries.
- `POST /search/hybrid`: Combines vector and graph search results with adaptive weighting.

### Data Management
- `POST /ingest/node`: Ingests a single document.
- `POST /ingest/bulk`: Ingests multiple documents in a batch.
- `GET /nodes`: Retrieves a paginated list of nodes.
- `GET /edges`: Retrieves a paginated list of relationships.

### System
- `GET /stats`: Returns system statistics and metrics.
- `GET /health`: Checks system health status.

## Project Structure
```
DevForge/
├── backend/
│   ├── app/
│   │   ├── api/            # API route handlers
│   │   ├── models/         # Data models and schemas
│   │   ├── services/       # Core business logic
│   │   └── main.py         # Application entry point
│   └── data/               # Data storage
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Application views
│   │   ├── lib/            # Utilities and API clients
│   │   └── types/          # TypeScript definitions
│   └── vite.config.ts      # Build configuration
└── README.md
```

## License
This project is licensed under the MIT License.
