"""
Search Request/Response Models

This module defines Pydantic models for search operations:
- Vector search (FAISS semantic similarity)
- Graph search (Neo4j traversal)
- Hybrid search (combined vector + graph)

Each search type has request and response models with proper validation
and documentation for OpenAPI spec generation.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from app.models.graph import Node


# =============================================================================
# Vector Search Models
# =============================================================================

class VectorSearchRequest(BaseModel):
    """
    Request model for vector-only semantic search.
    
    Uses FAISS to find nodes with embeddings most similar to the query.
    
    Example:
        {
            "query_text": "machine learning in medical diagnosis",
            "top_k": 5
        }
    """
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The search query text"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    # Optional filters
    source_filter: Optional[str] = Field(
        default=None,
        description="Filter results by source metadata"
    )
    topic_filter: Optional[str] = Field(
        default=None,
        description="Filter results by topic metadata"
    )


class VectorSearchResult(BaseModel):
    """
    Single result from vector search.
    
    Attributes:
        node: The matching node
        cosine_similarity: Similarity score between query and node (0.0 to 1.0)
        rank: Position in the result list (1-indexed)
    """
    node: Node = Field(..., description="The matching node")
    cosine_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score"
    )
    rank: int = Field(..., ge=1, description="Position in results (1-indexed)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "node": {
                    "id": "node_abc123",
                    "text": "Deep learning for medical imaging",
                    "metadata": {"source": "paper", "topic": "healthcare"}
                },
                "cosine_similarity": 0.89,
                "rank": 1
            }
        }


class VectorSearchResponse(BaseModel):
    """
    Response model for vector search.
    
    Contains the list of results and metadata about the search.
    """
    query_text: str = Field(..., description="The original search query")
    results: list[VectorSearchResult] = Field(
        default_factory=list,
        description="List of search results ordered by similarity"
    )
    total_results: int = Field(..., description="Number of results returned")
    search_time_ms: float = Field(
        default=0.0,
        description="Search execution time in milliseconds"
    )


# =============================================================================
# Graph Search Models
# =============================================================================

class GraphSearchRequest(BaseModel):
    """
    Request model for graph traversal search.
    
    Performs multi-hop traversal from a starting node using Neo4j.
    
    Example:
        {
            "start_id": "node_abc123",
            "depth": 2,
            "include_edges": true
        }
    """
    start_id: str = Field(
        ...,
        description="ID of the starting node for traversal"
    )
    depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum traversal depth (hops)"
    )
    include_edges: bool = Field(
        default=False,
        description="Whether to include edge information in results"
    )
    edge_type_filter: Optional[str] = Field(
        default=None,
        description="Filter traversal to specific edge type"
    )


class GraphTraversalNode(BaseModel):
    """
    Node information from graph traversal with hop distance.
    """
    node: Node = Field(..., description="The traversed node")
    hop_distance: int = Field(
        ...,
        ge=0,
        description="Number of hops from the start node"
    )
    path_weight: float = Field(
        default=0.0,
        description="Cumulative weight along the path"
    )


class GraphSearchResponse(BaseModel):
    """
    Response model for graph traversal search.
    
    Returns all nodes reachable within the specified depth.
    """
    start_node: Node = Field(..., description="The starting node")
    traversed_nodes: list[GraphTraversalNode] = Field(
        default_factory=list,
        description="Nodes found during traversal"
    )
    total_nodes: int = Field(..., description="Total number of nodes found")
    max_depth_reached: int = Field(
        ...,
        description="Maximum depth actually reached"
    )


# =============================================================================
# Hybrid Search Models
# =============================================================================

class HybridSearchRequest(BaseModel):
    """
    Request model for hybrid search combining vector and graph signals.
    
    The hybrid search:
    1. Gets semantic candidates from FAISS
    2. Computes graph-based signals from Neo4j
    3. Combines scores using adaptive or custom weights
    
    Example:
        {
            "query_text": "AI applications in healthcare",
            "top_k": 5,
            "vector_weight": 0.7,
            "graph_weight": 0.3
        }
    """
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The search query text"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of final results to return"
    )
    candidate_k: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Number of candidates to fetch from vector search before re-ranking"
    )
    vector_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity (0-1). If null, uses adaptive weighting."
    )
    graph_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for graph score (0-1). If null, uses adaptive weighting."
    )
    # Optional filters
    source_filter: Optional[str] = Field(
        default=None,
        description="Filter results by source metadata"
    )
    topic_filter: Optional[str] = Field(
        default=None,
        description="Filter results by topic metadata"
    )


class HybridSearchResult(BaseModel):
    """
    Single result from hybrid search with detailed scoring breakdown.
    
    This model shows how vector and graph signals were combined,
    demonstrating the value of hybrid search over vector-only.
    
    Attributes:
        node: The matching node
        cosine_similarity: Raw cosine similarity from FAISS (0.0 to 1.0)
        cosine_normalized: Normalized cosine score among candidates (0.0 to 1.0)
        graph_score: Raw graph-based score from Neo4j
        graph_normalized: Normalized graph score among candidates (0.0 to 1.0)
        final_score: Combined score: α * cosine_norm + β * graph_norm
        rank: Position in the result list (1-indexed)
        vector_only_rank: What rank this would have with vector search only
    """
    node: Node = Field(..., description="The matching node")
    
    # Vector score components
    cosine_similarity: float = Field(
        ...,
        description="Raw cosine similarity score"
    )
    cosine_normalized: float = Field(
        ...,
        description="Normalized cosine score (0-1)"
    )
    
    # Graph score components
    graph_score: float = Field(
        ...,
        description="Raw graph-based score"
    )
    graph_normalized: float = Field(
        ...,
        description="Normalized graph score (0-1)"
    )
    
    # Combined score
    final_score: float = Field(
        ...,
        description="Final combined score"
    )
    
    # Ranking info
    rank: int = Field(..., ge=1, description="Hybrid search rank")
    vector_only_rank: int = Field(
        ...,
        ge=1,
        description="Rank if using vector search only (for comparison)"
    )
    
    # Graph metadata
    degree: int = Field(
        default=0,
        description="Number of edges connected to this node"
    )
    avg_edge_weight: float = Field(
        default=0.0,
        description="Average weight of connected edges"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "node": {
                    "id": "node_abc123",
                    "text": "AI in healthcare: A comprehensive review",
                    "metadata": {"source": "paper", "topic": "healthcare"}
                },
                "cosine_similarity": 0.85,
                "cosine_normalized": 0.92,
                "graph_score": 1.73,
                "graph_normalized": 0.78,
                "final_score": 0.878,
                "rank": 1,
                "vector_only_rank": 3,
                "degree": 5,
                "avg_edge_weight": 0.7
            }
        }


class HybridSearchResponse(BaseModel):
    """
    Response model for hybrid search.
    
    Includes detailed information about the search process and
    how vector and graph signals were combined.
    """
    query_text: str = Field(..., description="The original search query")
    results: list[HybridSearchResult] = Field(
        default_factory=list,
        description="List of search results ordered by final score"
    )
    total_results: int = Field(..., description="Number of results returned")
    
    # Weight information
    vector_weight_used: float = Field(
        ...,
        description="Vector weight used (α)"
    )
    graph_weight_used: float = Field(
        ...,
        description="Graph weight used (β)"
    )
    weights_adaptive: bool = Field(
        ...,
        description="Whether adaptive weights were used"
    )
    
    # Performance metrics
    search_time_ms: float = Field(
        default=0.0,
        description="Total search execution time in milliseconds"
    )
    
    # Comparison info
    ranking_changed: bool = Field(
        default=False,
        description="Whether hybrid ranking differs from vector-only ranking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "AI in healthcare",
                "total_results": 5,
                "vector_weight_used": 0.7,
                "graph_weight_used": 0.3,
                "weights_adaptive": True,
                "search_time_ms": 45.2,
                "ranking_changed": True,
                "results": []
            }
        }


# =============================================================================
# Pagination Models
# =============================================================================

class PaginationParams(BaseModel):
    """
    Pagination parameters for list endpoints.
    """
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of items per page"
    )


class PaginatedResponse(BaseModel):
    """
    Generic paginated response wrapper.
    """
    items: list[Any] = Field(default_factory=list, description="Page items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there's a next page")
    has_prev: bool = Field(..., description="Whether there's a previous page")
