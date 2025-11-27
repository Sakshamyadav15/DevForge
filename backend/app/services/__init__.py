"""
Services package.

Contains core business logic:
- EmbeddingService: Text to vector embeddings
- VectorStore: FAISS index management
- GraphStore: Neo4j database operations
- SnapshotManager: JSON persistence
- HybridEngine: Combined vector + graph retrieval
"""

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.snapshot import SnapshotManager
from app.services.hybrid_engine import HybridEngine

__all__ = [
    "EmbeddingService",
    "VectorStore",
    "GraphStore",
    "SnapshotManager",
    "HybridEngine",
]
