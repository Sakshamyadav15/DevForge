"""
Application Configuration

This module defines all configuration settings for the DevForge backend service.
Settings can be overridden via environment variables for different deployment environments.

Configuration categories:
- Neo4j connection settings
- FAISS/Vector store settings
- Embedding model settings
- Snapshot persistence settings
- API settings
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden by setting environment variables with the
    same name (case-insensitive). For example:
    - NEO4J_URI=bolt://localhost:7687
    - USE_MOCK_EMBEDDINGS=true
    """
    
    # ==========================================================================
    # Neo4j Configuration
    # ==========================================================================
    # Neo4j connection URI - default assumes local instance
    neo4j_uri: str = "bolt://localhost:7687"
    # Neo4j authentication credentials
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    # Database name (use "neo4j" for Community Edition)
    neo4j_database: str = "neo4j"
    
    # ==========================================================================
    # Embedding Model Configuration
    # ==========================================================================
    # The embedding model to use - all-MiniLM-L6-v2 is chosen because:
    # 1. Good balance of speed and accuracy
    # 2. CPU-friendly (no GPU required)
    # 3. Produces 384-dimensional embeddings (compact but expressive)
    # 4. Well-suited for semantic similarity tasks
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Embedding dimension (must match the model output)
    embedding_dimension: int = 384
    # Whether to use mock embeddings (useful for testing/offline environments)
    use_mock_embeddings: bool = False
    
    # ==========================================================================
    # Snapshot/Persistence Configuration
    # ==========================================================================
    # Path to the JSON snapshot file - this is the ground truth for all data
    # Neo4j and FAISS are rebuilt from this snapshot on startup
    snapshot_path: str = "data/snapshot.json"
    # Whether to auto-rebuild indexes from snapshot on startup
    auto_rebuild_on_startup: bool = True
    
    # ==========================================================================
    # FAISS Vector Store Configuration
    # ==========================================================================
    # Use inner product for cosine similarity (vectors must be normalized)
    faiss_use_inner_product: bool = True
    
    # ==========================================================================
    # Edge Type Weights for Graph Scoring
    # ==========================================================================
    # These weights influence how different relationship types affect
    # the graph-based scoring in hybrid search
    # Higher weight = more importance in ranking
    edge_type_weights: dict[str, float] = {
        "RELATED_TO": 0.5,      # General relationship
        "SUPPORTS": 0.7,        # Supporting evidence
        "CITES": 1.0,           # Direct citation (strong link)
        "BELONGS_TO_TOPIC": 0.4, # Topic membership
        "CONTRADICTS": 0.3,     # Opposing viewpoint
        "EXTENDS": 0.8,         # Extension/elaboration
    }
    
    # ==========================================================================
    # API Configuration
    # ==========================================================================
    # API title and description for OpenAPI docs
    api_title: str = "DevForge - Hybrid Vector + Graph Database API"
    api_description: str = """
    A hybrid retrieval system combining:
    - **FAISS** for fast vector similarity search
    - **Neo4j** for graph-based reasoning and multi-hop traversal
    - **Hybrid search** that adaptively combines both approaches
    
    ## Features
    - Node and Edge CRUD operations
    - Vector-only semantic search
    - Graph traversal with configurable depth
    - Hybrid search with adaptive weighting
    - Local JSON snapshot persistence
    """
    api_version: str = "1.0.0"
    
    # Default pagination settings
    default_page_size: int = 10
    max_page_size: int = 100
    
    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    log_level: str = "INFO"
    
    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance - import this in other modules
settings = Settings()


def get_data_dir() -> Path:
    """
    Get the data directory path, creating it if it doesn't exist.
    
    Returns:
        Path object pointing to the data directory
    """
    data_dir = Path(settings.snapshot_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_snapshot_path() -> Path:
    """
    Get the full path to the snapshot file.
    
    Returns:
        Path object pointing to the snapshot JSON file
    """
    return Path(settings.snapshot_path)
