"""
DevForge - Hybrid Vector + Graph Database API

Main FastAPI application entry point.

This module:
- Initializes all services (embedding, vector store, graph store, snapshot)
- Wires up API routers
- Handles startup/shutdown events
- Exposes OpenAPI documentation

To run the server:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - OpenAPI JSON: http://localhost:8000/openapi.json
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.snapshot import SnapshotManager
from app.services.hybrid_engine import HybridEngine
from app.api import nodes_router, edges_router, search_router

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Service Instances
# =============================================================================
# These are initialized during startup and used by API route dependencies

embedding_service: EmbeddingService = None
vector_store: VectorStore = None
graph_store: GraphStore = None
snapshot_manager: SnapshotManager = None
hybrid_engine: HybridEngine = None


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize services, connect to Neo4j, rebuild indexes
    - Shutdown: Close connections, cleanup resources
    """
    global embedding_service, vector_store, graph_store, snapshot_manager, hybrid_engine
    
    logger.info("=" * 60)
    logger.info("DevForge starting up...")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Initialize Services
    # -------------------------------------------------------------------------
    
    # 1. Initialize Embedding Service
    logger.info("Initializing embedding service...")
    embedding_service = EmbeddingService(use_mock=settings.use_mock_embeddings)
    logger.info(f"Embedding service: {embedding_service}")
    
    # 2. Initialize Vector Store (FAISS)
    logger.info("Initializing vector store...")
    vector_store = VectorStore(dim=embedding_service.dimension)
    
    # Try to load persisted FAISS index
    faiss_path = "data/vector_index"
    faiss_loaded = vector_store.load(faiss_path)
    if faiss_loaded:
        logger.info(f"Loaded persisted FAISS index: {vector_store}")
    else:
        logger.info(f"No persisted FAISS index found")
    
    # 3. Initialize Graph Store (Neo4j)
    logger.info("Connecting to Neo4j...")
    neo4j_has_data = False
    try:
        graph_store = GraphStore()
        logger.info("Neo4j connected successfully")
        neo4j_node_count = graph_store.get_node_count()
        if neo4j_node_count > 0:
            neo4j_has_data = True
            logger.info(f"Neo4j already has {neo4j_node_count} nodes")
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j: {e}")
        logger.warning("Running in degraded mode - graph operations will fail")
        graph_store = None
    
    # 4. Initialize Snapshot Manager
    logger.info("Initializing snapshot manager...")
    snapshot_manager = SnapshotManager()
    
    # 5. Rebuild indexes from snapshot if needed
    # Only rebuild if BOTH FAISS and Neo4j are empty
    needs_rebuild = not faiss_loaded or not neo4j_has_data
    
    if settings.auto_rebuild_on_startup and graph_store is not None and needs_rebuild:
        logger.info("Rebuilding indexes from snapshot (this may take a while for large datasets)...")
        try:
            nodes_rebuilt, edges_rebuilt = snapshot_manager.rebuild_indexes(
                graph_store=graph_store,
                vector_store=vector_store,
                embedding_service=embedding_service
            )
            logger.info(f"Rebuilt {nodes_rebuilt} nodes and {edges_rebuilt} edges from snapshot")
            
            # Save FAISS index for next startup
            vector_store.save(faiss_path)
        except Exception as e:
            logger.warning(f"Failed to rebuild indexes: {e}")
    
    # 6. Initialize Hybrid Engine
    if graph_store is not None:
        hybrid_engine = HybridEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_service=embedding_service
        )
        logger.info("Hybrid engine initialized")
    else:
        hybrid_engine = None
        logger.warning("Hybrid engine not available (Neo4j not connected)")
    
    logger.info("=" * 60)
    logger.info("DevForge startup complete!")
    logger.info("=" * 60)
    
    # Yield control to the application
    yield
    
    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------
    logger.info("DevForge shutting down...")
    
    if graph_store is not None:
        graph_store.close()
        logger.info("Neo4j connection closed")
    
    logger.info("DevForge shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# =============================================================================
# CORS Middleware
# =============================================================================
# Enable CORS for development (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Routers
# =============================================================================
app.include_router(nodes_router)
app.include_router(edges_router)
app.include_router(search_router)


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - basic API information.
    """
    return {
        "name": "DevForge",
        "description": "Hybrid Vector + Graph Database for AI Retrieval",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all system components.
    """
    health_status = {
        "status": "healthy",
        "components": {
            "embedding_service": {
                "status": "healthy" if embedding_service else "unavailable",
                "using_mock": embedding_service.is_mock if embedding_service else None,
                "dimension": embedding_service.dimension if embedding_service else None
            },
            "vector_store": {
                "status": "healthy" if vector_store else "unavailable",
                "size": len(vector_store) if vector_store else 0
            },
            "graph_store": {
                "status": "healthy" if graph_store else "unavailable"
            },
            "snapshot_manager": {
                "status": "healthy" if snapshot_manager else "unavailable",
                "node_count": snapshot_manager.get_node_count() if snapshot_manager else 0,
                "edge_count": snapshot_manager.get_edge_count() if snapshot_manager else 0
            },
            "hybrid_engine": {
                "status": "healthy" if hybrid_engine else "unavailable"
            }
        }
    }
    
    # Overall status is unhealthy if any critical component is down
    if not graph_store:
        health_status["status"] = "degraded"
        health_status["message"] = "Neo4j not connected - graph operations unavailable"
    
    return health_status


@app.get("/healthz", tags=["Health"])
async def health_check_simple():
    """
    Simple health check endpoint (frontend-compatible).
    
    Returns a simple status for load balancers and frontends.
    """
    if embedding_service and vector_store and snapshot_manager:
        return {"status": "ok"}
    return {"status": "error", "message": "One or more services unavailable"}


@app.get("/stats", tags=["Health"])
async def get_stats():
    """
    Get system statistics (frontend-compatible format).
    
    Returns counts and metrics for all data stores.
    """
    from datetime import datetime
    
    # Get snapshot last modified time
    snapshot_updated = None
    if snapshot_manager and snapshot_manager.snapshot_path.exists():
        mtime = snapshot_manager.snapshot_path.stat().st_mtime
        snapshot_updated = datetime.fromtimestamp(mtime).isoformat()
    
    return {
        # Frontend-compatible fields
        "nodes": snapshot_manager.get_node_count() if snapshot_manager else 0,
        "edges": snapshot_manager.get_edge_count() if snapshot_manager else 0,
        "snapshot_last_updated": snapshot_updated,
        "vector_index_size": len(vector_store) if vector_store else 0,
        # Extended info
        "embedding": {
            "model": settings.embedding_model_name if not embedding_service.is_mock else "mock",
            "dimension": embedding_service.dimension if embedding_service else 0
        },
        "config": {
            "neo4j_uri": settings.neo4j_uri,
            "auto_rebuild": settings.auto_rebuild_on_startup,
            "log_level": settings.log_level
        }
    }


@app.post("/admin/save-index", tags=["Admin"])
async def save_faiss_index():
    """
    Save the current FAISS index to disk.
    
    Call this after ingesting new data to persist the index.
    """
    try:
        faiss_path = "data/vector_index"
        vector_store.save(faiss_path)
        return {
            "status": "success",
            "message": f"FAISS index saved ({vector_store.size} vectors)",
            "path": faiss_path
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/admin/rebuild", tags=["Admin"])
async def force_rebuild():
    """
    Force rebuild of FAISS and Neo4j indexes from snapshot.
    
    WARNING: This will clear all existing data and regenerate from snapshot.
    This can take a long time for large datasets.
    """
    if graph_store is None:
        return {"status": "error", "message": "Neo4j not connected"}
    
    try:
        nodes_rebuilt, edges_rebuilt = snapshot_manager.rebuild_indexes(
            graph_store=graph_store,
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        
        # Save FAISS index
        faiss_path = "data/vector_index"
        vector_store.save(faiss_path)
        
        return {
            "status": "success",
            "nodes_rebuilt": nodes_rebuilt,
            "edges_rebuilt": edges_rebuilt,
            "faiss_saved": True
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
