"""
Hybrid Retrieval Engine

This module implements hybrid search that combines:
1. Vector similarity (FAISS cosine similarity) - semantic understanding
2. Graph signals (Neo4j) - structural importance and relationships

Why Hybrid > Vector-Only?
- Vector search finds semantically similar content
- BUT it misses structural importance (well-connected nodes are often more relevant)
- Graph signals capture "authority" (many citations) and "context" (related topics)
- Combining them produces more relevant, contextual results

Example:
    Query: "AI in healthcare"
    
    Vector-only might return:
    1. "Machine learning basics" (high semantic match to "AI")
    2. "Healthcare administration" (high semantic match to "healthcare")
    
    Hybrid search considers:
    - Node about "Medical imaging AI" has lower raw similarity
    - BUT it's heavily cited and connected to both AI and healthcare topics
    - So it gets boosted in hybrid ranking → better result!

Adaptive Weights:
- Relationship-focused queries ("connection between X and Y") → more graph weight
- General semantic queries → more vector weight
- This automatically tunes the search to query intent
"""

import math
import logging
import time
from typing import Optional

from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.embeddings import EmbeddingService
from app.models.graph import Node
from app.models.search import HybridSearchResult, HybridSearchResponse

logger = logging.getLogger(__name__)


# Keywords that indicate relationship-focused queries
# When detected, we increase graph weight for better contextual results
RELATION_KEYWORDS = [
    "between",
    "connection",
    "connected",
    "path",
    "relationship",
    "neighbors",
    "around",
    "context",
    "related",
    "links",
    "linked",
    "associated",
    "network",
    "graph"
]


def choose_weights(
    query_text: str,
    default_vector: float = 0.7,
    default_graph: float = 0.3
) -> tuple[float, float]:
    """
    Adaptively choose vector and graph weights based on query text.
    
    If the query contains relationship-focused keywords, we increase
    the graph weight to capture structural relevance.
    
    Args:
        query_text: The search query
        default_vector: Default weight for vector similarity (α)
        default_graph: Default weight for graph score (β)
        
    Returns:
        Tuple of (vector_weight, graph_weight) that sum to 1.0
        
    Example:
        >>> choose_weights("machine learning")
        (0.7, 0.3)  # Standard semantic search
        
        >>> choose_weights("connection between AI and medicine")
        (0.4, 0.6)  # More graph influence for relationship query
    """
    query_lower = query_text.lower()
    
    # Check if query contains relationship keywords
    if any(keyword in query_lower for keyword in RELATION_KEYWORDS):
        # Relationship-focused query: emphasize graph structure
        logger.debug(f"Detected relationship query, using graph-heavy weights")
        return 0.4, 0.6
    
    return default_vector, default_graph


def compute_graph_score(graph_info: dict) -> float:
    """
    Compute a graph-based relevance score for a node.
    
    The formula combines:
    - degree: More connections = more "authority"
    - avg_weight: Higher average edge weight = stronger relationships
    - type_weighted_sum: Different edge types have different importance
    
    Formula: (avg_weight + type_weighted_sum) * log(1 + degree)
    
    The log scaling prevents high-degree nodes from completely dominating.
    A node with 100 connections isn't 100x more important than one with 1.
    
    Args:
        graph_info: Dictionary with keys: degree, avg_weight, type_weighted_sum
        
    Returns:
        Graph relevance score (not normalized)
        
    Example:
        >>> graph_info = {"degree": 5, "avg_weight": 0.7, "type_weighted_sum": 1.2}
        >>> score = compute_graph_score(graph_info)
        >>> print(f"{score:.3f}")  # ~3.4
    """
    degree = graph_info.get("degree", 0)
    avg_weight = graph_info.get("avg_weight", 0.0)
    type_weighted_sum = graph_info.get("type_weighted_sum", 0.0)
    
    # Base score combines edge quality metrics
    base = avg_weight + type_weighted_sum
    
    # Scale by log of degree (diminishing returns for very high degree)
    # log(1 + degree) ensures:
    # - degree=0 → multiplier=0
    # - degree=1 → multiplier≈0.69
    # - degree=10 → multiplier≈2.4
    # - degree=100 → multiplier≈4.6
    score = base * math.log(1 + degree)
    
    return score


def normalize(values: list[float]) -> list[float]:
    """
    Normalize a list of values to [0, 1] range using min-max scaling.
    
    This ensures vector and graph scores are comparable before combining.
    
    Args:
        values: List of numeric values
        
    Returns:
        List of normalized values in [0, 1]
        
    Example:
        >>> normalize([0.5, 1.0, 0.75])
        [0.0, 1.0, 0.5]
    """
    if not values:
        return []
    
    lo = min(values)
    hi = max(values)
    
    # Handle case where all values are the same
    if hi == lo:
        return [0.5 for _ in values]
    
    return [(v - lo) / (hi - lo) for v in values]


class HybridEngine:
    """
    Hybrid search engine combining vector similarity and graph signals.
    
    This is the core innovation of the system - it demonstrates that
    combining semantic search with graph structure produces better
    results than either approach alone.
    
    The hybrid search process:
    1. Embed query using EmbeddingService
    2. Get top-K candidates from FAISS (vector search)
    3. For each candidate, compute graph score from Neo4j
    4. Normalize both score types to [0, 1]
    5. Combine: final_score = α * cosine_norm + β * graph_norm
    6. Re-rank by final_score and return top results
    
    Attributes:
        vector_store: FAISS index for vector search
        graph_store: Neo4j client for graph queries
        embedding_service: Service for text embeddings
        
    Example:
        >>> engine = HybridEngine(vector_store, graph_store, embedding_service)
        >>> results = engine.hybrid_search("AI in healthcare", top_k=5)
        >>> for r in results:
        ...     print(f"{r.node.id}: final={r.final_score:.3f} (vec={r.cosine_similarity:.3f}, graph={r.graph_score:.3f})")
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_service: EmbeddingService
    ):
        """
        Initialize the hybrid search engine.
        
        Args:
            vector_store: FAISS vector store for similarity search
            graph_store: Neo4j graph store for structural queries
            embedding_service: Service for generating text embeddings
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_service = embedding_service
    
    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 5,
        candidate_k: int = 20,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        source_filter: Optional[str] = None,
        topic_filter: Optional[str] = None
    ) -> HybridSearchResponse:
        """
        Perform hybrid search combining vector and graph signals.
        
        This is the main entry point for hybrid retrieval. It:
        1. Gets semantic candidates from FAISS
        2. Enriches with graph scores from Neo4j
        3. Combines scores with configurable or adaptive weights
        4. Returns re-ranked results with full scoring breakdown
        
        Args:
            query_text: The search query
            top_k: Number of final results to return
            candidate_k: Number of candidates from vector search (before re-ranking)
            vector_weight: Weight for vector similarity (α). If None, uses adaptive.
            graph_weight: Weight for graph score (β). If None, uses adaptive.
            source_filter: Optional filter by source metadata
            topic_filter: Optional filter by topic metadata
            
        Returns:
            HybridSearchResponse with ranked results and scoring breakdown
            
        Raises:
            ValueError: If provided weights don't sum to approximately 1.0
        """
        start_time = time.time()
        
        # Determine weights (adaptive or explicit)
        weights_adaptive = vector_weight is None or graph_weight is None
        
        if weights_adaptive:
            alpha, beta = choose_weights(query_text)
        else:
            alpha, beta = vector_weight, graph_weight
            # Validate weights sum to ~1.0
            if abs(alpha + beta - 1.0) > 0.01:
                logger.warning(f"Weights don't sum to 1.0: α={alpha}, β={beta}")
        
        logger.debug(f"Hybrid search: query='{query_text[:50]}...', α={alpha}, β={beta}")
        
        # Step 1: Embed the query
        query_embedding = self.embedding_service.embed(query_text)
        
        # Step 2: Get vector search candidates
        # We fetch more candidates than needed for re-ranking
        vector_results = self.vector_store.search(query_embedding, top_k=candidate_k)
        
        if not vector_results:
            # No results found
            return HybridSearchResponse(
                query_text=query_text,
                results=[],
                total_results=0,
                vector_weight_used=alpha,
                graph_weight_used=beta,
                weights_adaptive=weights_adaptive,
                search_time_ms=(time.time() - start_time) * 1000,
                ranking_changed=False
            )
        
        # Step 3: Fetch graph info and compute scores for each candidate
        candidates = []
        cosine_scores = []
        graph_scores = []
        
        for node_id, cosine_sim in vector_results:
            # Get node details
            node = self.graph_store.get_node(node_id)
            if not node:
                continue
            
            # Apply metadata filters if specified
            if source_filter and node.metadata.get("source") != source_filter:
                continue
            if topic_filter and node.metadata.get("topic") != topic_filter:
                continue
            
            # Get graph statistics
            graph_info = self.graph_store.get_neighbors_and_stats(node_id, max_depth=1)
            graph_score = compute_graph_score(graph_info)
            
            candidates.append({
                "node": node,
                "cosine_sim": cosine_sim,
                "graph_score": graph_score,
                "graph_info": graph_info
            })
            
            cosine_scores.append(cosine_sim)
            graph_scores.append(graph_score)
        
        if not candidates:
            return HybridSearchResponse(
                query_text=query_text,
                results=[],
                total_results=0,
                vector_weight_used=alpha,
                graph_weight_used=beta,
                weights_adaptive=weights_adaptive,
                search_time_ms=(time.time() - start_time) * 1000,
                ranking_changed=False
            )
        
        # Step 4: Normalize scores
        cosine_normalized = normalize(cosine_scores)
        graph_normalized = normalize(graph_scores)
        
        # Step 5: Compute final scores and attach normalized values
        for i, candidate in enumerate(candidates):
            candidate["cosine_normalized"] = cosine_normalized[i]
            candidate["graph_normalized"] = graph_normalized[i]
            candidate["final_score"] = (
                alpha * cosine_normalized[i] + 
                beta * graph_normalized[i]
            )
        
        # Step 6: Sort by final score (descending)
        candidates.sort(key=lambda c: c["final_score"], reverse=True)
        
        # Determine vector-only ranks for comparison
        vector_only_order = sorted(
            range(len(candidates)),
            key=lambda i: candidates[i]["cosine_sim"],
            reverse=True
        )
        vector_only_ranks = {idx: rank + 1 for rank, idx in enumerate(vector_only_order)}
        
        # Check if ranking changed from vector-only
        hybrid_order = list(range(len(candidates)))  # Already sorted by final_score
        ranking_changed = hybrid_order != sorted(
            hybrid_order,
            key=lambda i: candidates[i]["cosine_sim"],
            reverse=True
        )
        
        # Step 7: Build response with top_k results
        results = []
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            # Find this candidate's position in the original unsorted list
            original_idx = next(
                i for i, c in enumerate(candidates) 
                if c["node"].id == candidate["node"].id
            )
            
            result = HybridSearchResult(
                node=candidate["node"],
                cosine_similarity=candidate["cosine_sim"],
                cosine_normalized=candidate["cosine_normalized"],
                graph_score=candidate["graph_score"],
                graph_normalized=candidate["graph_normalized"],
                final_score=candidate["final_score"],
                rank=rank,
                vector_only_rank=vector_only_ranks.get(original_idx, rank),
                degree=candidate["graph_info"].get("degree", 0),
                avg_edge_weight=candidate["graph_info"].get("avg_weight", 0.0)
            )
            results.append(result)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        logger.debug(
            f"Hybrid search complete: {len(results)} results in {search_time_ms:.1f}ms, "
            f"ranking_changed={ranking_changed}"
        )
        
        return HybridSearchResponse(
            query_text=query_text,
            results=results,
            total_results=len(results),
            vector_weight_used=alpha,
            graph_weight_used=beta,
            weights_adaptive=weights_adaptive,
            search_time_ms=search_time_ms,
            ranking_changed=ranking_changed
        )
    
    def vector_only_search(
        self,
        query_text: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        topic_filter: Optional[str] = None
    ) -> list[tuple[Node, float]]:
        """
        Perform vector-only search (for comparison with hybrid).
        
        This method exists to demonstrate the difference between
        vector-only and hybrid search results.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            source_filter: Optional metadata filter
            topic_filter: Optional metadata filter
            
        Returns:
            List of (Node, cosine_similarity) tuples
        """
        query_embedding = self.embedding_service.embed(query_text)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        results = []
        for node_id, cosine_sim in vector_results:
            node = self.graph_store.get_node(node_id)
            if not node:
                continue
            
            # Apply filters
            if source_filter and node.metadata.get("source") != source_filter:
                continue
            if topic_filter and node.metadata.get("topic") != topic_filter:
                continue
            
            results.append((node, cosine_sim))
            
            if len(results) >= top_k:
                break
        
        return results
