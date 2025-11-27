"""
Search API Routes

This module provides REST API endpoints for search operations:
- POST /search/vector - Vector-only semantic search
- GET /search/graph - Graph traversal (multi-hop reasoning)
- POST /search/hybrid - Hybrid search combining vector + graph

The hybrid search endpoint demonstrates the key value proposition:
combining semantic similarity with graph structure produces more
relevant results than either approach alone.
"""

import logging
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from app.models.graph import Node
from app.models.search import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
    GraphSearchRequest,
    GraphSearchResponse,
    GraphTraversalNode,
    HybridSearchRequest,
    HybridSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


# Dependency injection for services
def get_graph_store():
    """Dependency to get GraphStore instance."""
    from app.main import graph_store
    return graph_store


def get_vector_store():
    """Dependency to get VectorStore instance."""
    from app.main import vector_store
    return vector_store


def get_embedding_service():
    """Dependency to get EmbeddingService instance."""
    from app.main import embedding_service
    return embedding_service


def get_hybrid_engine():
    """Dependency to get HybridEngine instance."""
    from app.main import hybrid_engine
    return hybrid_engine


@router.post(
    "/vector",
    response_model=VectorSearchResponse,
    summary="Vector-only semantic search",
    description="""
    Perform semantic search using FAISS vector similarity.
    
    This endpoint:
    1. Embeds the query text using the embedding model
    2. Searches FAISS for the most similar vectors
    3. Returns nodes ranked by cosine similarity
    
    Use this for pure semantic search without considering graph structure.
    Compare with /search/hybrid to see the benefit of graph signals.
    """
)
async def vector_search(
    request: VectorSearchRequest,
    graph_store=Depends(get_graph_store),
    vector_store=Depends(get_vector_store),
    embedding_service=Depends(get_embedding_service)
) -> VectorSearchResponse:
    """
    Perform vector-only semantic search.
    
    Args:
        request: Search request with query_text and top_k
        
    Returns:
        VectorSearchResponse with ranked results
    """
    start_time = time.time()
    
    try:
        # Embed the query
        query_embedding = embedding_service.embed(request.query_text)
        
        # Search FAISS
        # Fetch extra candidates for filtering
        search_k = request.top_k * 2 if request.source_filter or request.topic_filter else request.top_k
        vector_results = vector_store.search(query_embedding, top_k=search_k)
        
        # Build results with node details
        results = []
        rank = 1
        
        for node_id, cosine_sim in vector_results:
            if len(results) >= request.top_k:
                break
            
            # Get node details from Neo4j
            node = graph_store.get_node(node_id)
            if not node:
                continue
            
            # Apply metadata filters
            if request.source_filter and node.metadata.get("source") != request.source_filter:
                continue
            if request.topic_filter and node.metadata.get("topic") != request.topic_filter:
                continue
            
            results.append(VectorSearchResult(
                node=node,
                cosine_similarity=cosine_sim,
                rank=rank
            ))
            rank += 1
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return VectorSearchResponse(
            query_text=request.query_text,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/graph",
    response_model=GraphSearchResponse,
    summary="Graph traversal search",
    description="""
    Perform multi-hop graph traversal from a starting node.
    
    This endpoint uses Neo4j to traverse the knowledge graph,
    returning all nodes reachable within the specified depth.
    
    This is the "multi-hop reasoning" capability - finding related
    information through chains of relationships.
    
    Example: Starting from "Machine Learning", with depth=2:
    - Hop 1: "Deep Learning", "Neural Networks" (directly connected)
    - Hop 2: "Computer Vision", "NLP" (connected to hop 1 nodes)
    """
)
async def graph_search(
    start_id: str = Query(..., description="ID of the starting node"),
    depth: int = Query(default=1, ge=1, le=5, description="Maximum traversal depth"),
    edge_type: Optional[str] = Query(default=None, description="Filter by edge type"),
    graph_store=Depends(get_graph_store)
) -> GraphSearchResponse:
    """
    Perform graph traversal from a starting node.
    
    Args:
        start_id: The node ID to start traversal from
        depth: Maximum number of hops (1-5)
        edge_type: Optional filter to traverse only certain edge types
        
    Returns:
        GraphSearchResponse with traversed nodes and distances
        
    Raises:
        HTTPException 404: If start node not found
    """
    try:
        # Verify start node exists
        start_node = graph_store.get_node(start_id)
        if not start_node:
            raise HTTPException(status_code=404, detail=f"Start node '{start_id}' not found")
        
        # Perform traversal with distances
        traversal_results = graph_store.traverse_with_distances(start_id, depth)
        
        # Build response
        traversed_nodes = []
        max_depth_reached = 0
        
        for node, hop_distance, path_weight in traversal_results:
            traversed_nodes.append(GraphTraversalNode(
                node=node,
                hop_distance=hop_distance,
                path_weight=path_weight
            ))
            max_depth_reached = max(max_depth_reached, hop_distance)
        
        # Sort by hop distance, then by path weight (descending)
        traversed_nodes.sort(key=lambda x: (x.hop_distance, -x.path_weight))
        
        return GraphSearchResponse(
            start_node=start_node,
            traversed_nodes=traversed_nodes,
            total_nodes=len(traversed_nodes),
            max_depth_reached=max_depth_reached
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    summary="Hybrid search (vector + graph)",
    description="""
    Perform hybrid search combining vector similarity and graph signals.
    
    This is the key innovation of the system. It demonstrates that
    combining semantic search with graph structure produces better
    results than either approach alone.
    
    **How it works:**
    1. Embed the query and get semantic candidates from FAISS
    2. For each candidate, compute graph-based scores from Neo4j
    3. Normalize both score types
    4. Combine: `final_score = α × cosine_norm + β × graph_norm`
    5. Re-rank and return top results
    
    **Adaptive Weights:**
    If vector_weight and graph_weight are not provided, the system
    automatically detects query intent:
    - Relationship queries ("connection between X and Y") → more graph weight
    - General queries → more vector weight
    
    **Response includes:**
    - Full scoring breakdown for each result
    - `vector_only_rank` to show how ranking changed from vector-only
    - `ranking_changed` flag indicating if hybrid improved ranking
    """
)
async def hybrid_search(
    request: HybridSearchRequest,
    hybrid_engine=Depends(get_hybrid_engine)
) -> HybridSearchResponse:
    """
    Perform hybrid search combining vector and graph signals.
    
    Args:
        request: HybridSearchRequest with query and optional weights
        
    Returns:
        HybridSearchResponse with detailed scoring breakdown
    """
    try:
        response = hybrid_engine.hybrid_search(
            query_text=request.query_text,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            vector_weight=request.vector_weight,
            graph_weight=request.graph_weight,
            source_filter=request.source_filter,
            topic_filter=request.topic_filter
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/hybrid/simple",
    summary="Hybrid search (frontend-compatible)",
    description="Simplified hybrid search endpoint matching frontend expectations. Falls back to vector-only search if Neo4j is unavailable."
)
async def hybrid_search_simple(
    request: dict,
    hybrid_engine=Depends(get_hybrid_engine),
    graph_store=Depends(get_graph_store),
    vector_store=Depends(get_vector_store),
    embedding_service=Depends(get_embedding_service)
):
    """
    Frontend-compatible hybrid search.
    
    Accepts: {query, vector_weight?, graph_weight?, top_k?, filters?: {topic?, category?}}
    Returns: {results: [{id, text_snippet, score, vector_score, graph_score, neighbors, metadata}]}
    
    Falls back to vector-only search if Neo4j/hybrid_engine is unavailable.
    """
    try:
        query = request.get("query") or request.get("query_text", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        top_k = request.get("top_k", 10)
        vector_weight = request.get("vector_weight", 0.7)
        graph_weight = request.get("graph_weight", 0.3)
        filters = request.get("filters", {})
        topic_filter = filters.get("topic")
        
        results = []
        
        # Use hybrid engine if available, otherwise fall back to vector-only
        if hybrid_engine is not None:
            response = hybrid_engine.hybrid_search(
                query_text=query,
                top_k=top_k,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
                topic_filter=topic_filter
            )
            
            # Transform to frontend format
            for r in response.results:
                # Get neighbor count
                edges = graph_store.get_edges_for_node(r.node.id) if graph_store else []
                neighbor_count = len(edges)
                
                results.append({
                    "id": r.node.id,
                    "title": r.node.metadata.get("title"),
                    "text_snippet": r.node.text[:300] + "..." if len(r.node.text) > 300 else r.node.text,
                    "score": round(r.final_score, 4),
                    "vector_score": round(r.cosine_similarity, 4),
                    "graph_score": round(r.graph_score, 4),
                    "neighbors": neighbor_count,
                    "metadata": r.node.metadata
                })
        else:
            # Fallback: Vector-only search using snapshot data
            from app.main import snapshot_manager
            
            # Embed query
            query_embedding = embedding_service.embed(query)
            
            # Search FAISS
            faiss_results = vector_store.search(query_embedding, top_k=top_k * 2)
            
            # Get node details from snapshot
            snapshot = snapshot_manager.load_snapshot()
            nodes_by_id = {n["id"]: n for n in snapshot.get("nodes", [])}
            
            for node_id, distance in faiss_results[:top_k]:
                node_data = nodes_by_id.get(node_id)
                if node_data:
                    # Apply topic filter if specified
                    if topic_filter and node_data.get("metadata", {}).get("topic") != topic_filter:
                        continue
                    
                    text = node_data.get("text", "")
                    cosine_sim = 1.0 - distance  # Convert distance to similarity
                    
                    results.append({
                        "id": node_id,
                        "title": node_data.get("metadata", {}).get("title"),
                        "text_snippet": text[:300] + "..." if len(text) > 300 else text,
                        "score": round(cosine_sim, 4),
                        "vector_score": round(cosine_sim, 4),
                        "graph_score": 0.0,  # No graph available
                        "neighbors": 0,
                        "metadata": node_data.get("metadata", {})
                    })
        
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/compare",
    summary="Compare vector vs hybrid search",
    description="""
    Compare vector-only and hybrid search results side-by-side.
    
    This endpoint is useful for demonstrating the value of hybrid search.
    It returns both vector-only and hybrid results for the same query,
    allowing you to see how graph signals change the ranking.
    """
)
async def compare_search(
    query_text: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
    graph_store=Depends(get_graph_store),
    vector_store=Depends(get_vector_store),
    embedding_service=Depends(get_embedding_service),
    hybrid_engine=Depends(get_hybrid_engine)
) -> dict:
    """
    Compare vector-only and hybrid search results.
    
    Args:
        query_text: The search query
        top_k: Number of results to return
        
    Returns:
        Dictionary with both vector and hybrid results
    """
    try:
        # Vector-only search
        query_embedding = embedding_service.embed(query_text)
        vector_results = vector_store.search(query_embedding, top_k=top_k)
        
        vector_nodes = []
        for node_id, cosine_sim in vector_results:
            node = graph_store.get_node(node_id)
            if node:
                vector_nodes.append({
                    "node_id": node.id,
                    "text": node.text[:100] + "..." if len(node.text) > 100 else node.text,
                    "cosine_similarity": round(cosine_sim, 4)
                })
        
        # Hybrid search
        hybrid_response = hybrid_engine.hybrid_search(
            query_text=query_text,
            top_k=top_k
        )
        
        hybrid_nodes = []
        for result in hybrid_response.results:
            hybrid_nodes.append({
                "node_id": result.node.id,
                "text": result.node.text[:100] + "..." if len(result.node.text) > 100 else result.node.text,
                "cosine_similarity": round(result.cosine_similarity, 4),
                "graph_score": round(result.graph_score, 4),
                "final_score": round(result.final_score, 4),
                "vector_only_rank": result.vector_only_rank,
                "hybrid_rank": result.rank
            })
        
        # Compute statistics
        ranking_changes = sum(
            1 for r in hybrid_response.results 
            if r.vector_only_rank != r.rank
        )
        
        return {
            "query": query_text,
            "vector_only_results": vector_nodes,
            "hybrid_results": hybrid_nodes,
            "weights_used": {
                "vector": hybrid_response.vector_weight_used,
                "graph": hybrid_response.graph_weight_used,
                "adaptive": hybrid_response.weights_adaptive
            },
            "statistics": {
                "ranking_changes": ranking_changes,
                "total_results": len(hybrid_nodes),
                "search_time_ms": round(hybrid_response.search_time_ms, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Compare search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
