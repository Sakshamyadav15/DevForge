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
    
    **Filtering:**
    - `source_filter`: Filter by source metadata
    - `topic_filter`: Filter by topic metadata
    - `metadata_filter`: Generic filter - only nodes matching ALL key-value pairs returned
      Example: `{"type": "note"}` returns only nodes where metadata.type = "note"
    
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
        request: Search request with query_text, top_k, and optional filters
        
    Returns:
        VectorSearchResponse with ranked results
    """
    start_time = time.time()
    
    try:
        # Embed the query
        query_embedding = embedding_service.embed(request.query_text)
        
        # Search FAISS
        # Fetch extra candidates for filtering
        has_filters = request.source_filter or request.topic_filter or request.metadata_filter
        search_k = request.top_k * 3 if has_filters else request.top_k
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
            
            # Apply generic metadata filter (TC-VEC-03)
            if request.metadata_filter:
                match = True
                for key, value in request.metadata_filter.items():
                    if node.metadata.get(key) != value:
                        match = False
                        break
                if not match:
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
    "/graph/traverse",
    response_model=GraphSearchResponse,
    summary="Graph traversal (GET) - BFS depth-limited",
    description="""
    Perform BFS (breadth-first) graph traversal from a starting node.
    
    Returns all nodes reachable within the specified depth, with their hop distances.
    
    **Example:**
    Given chain A -> B -> C -> D, query with `start_id=A, depth=2`:
    - Returns B (depth 1), C (depth 2)
    - Does NOT return D (depth 3, exceeds limit)
    
    **Cycle Handling:** Nodes are visited only once; traversal terminates safely.
    
    **Edge Type Filtering:** Use `edge_type` to filter traversal to specific relationship types.
    """
)
async def graph_traverse_get(
    start_id: str = Query(..., description="ID of the starting node"),
    depth: int = Query(default=2, ge=1, le=5, description="Maximum traversal depth (hops)"),
    edge_type: Optional[str] = Query(default=None, description="Filter by edge type (e.g., 'RELATED_TO', 'author_of')"),
    graph_store=Depends(get_graph_store)
) -> GraphSearchResponse:
    """
    GET endpoint for graph traversal (TC-GRAPH-01 compliant).
    
    Args:
        start_id: ID of the starting node
        depth: Maximum number of hops
        edge_type: Optional edge type filter
        
    Returns:
        GraphSearchResponse with traversed nodes and their hop distances
    """
    try:
        # Verify start node exists
        start_node = graph_store.get_node(start_id)
        if not start_node:
            raise HTTPException(status_code=404, detail=f"Start node '{start_id}' not found")
        
        # Perform traversal with distances
        traversal_results = graph_store.traverse_with_distances(start_id, depth, edge_type)
        
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
        logger.error(f"Graph traversal failed: {e}")
        raise HTTPException(status_code=500, detail=f"Traversal failed: {str(e)}")


@router.post(
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
    request: GraphSearchRequest,
    graph_store=Depends(get_graph_store)
) -> GraphSearchResponse:
    """
    Perform graph traversal from a starting node.
    
    Args:
        request: GraphSearchRequest with start_id, depth, edge_type_filter
        
    Returns:
        GraphSearchResponse with traversed nodes and distances
        
    Raises:
        HTTPException 404: If start node not found
    """
    try:
        start_id = request.start_id
        depth = request.depth or 1
        edge_type_filter = request.edge_type_filter
        
        # Verify start node exists
        start_node = graph_store.get_node(start_id)
        if not start_node:
            raise HTTPException(status_code=404, detail=f"Start node '{start_id}' not found")
        
        # Perform traversal with distances (supports edge type filtering)
        traversal_results = graph_store.traverse_with_distances(start_id, depth, edge_type_filter)
        
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


# =============================================================================
# Graph-Only Search Options (without needing to know a node ID)
# =============================================================================

@router.get(
    "/graph/by-topic",
    summary="Find nodes by topic and explore",
    description="""
    Find nodes by topic, then optionally explore their graph neighborhood.
    This is useful when you want to explore a topic area without knowing specific node IDs.
    """
)
async def graph_search_by_topic(
    topic: str = Query(..., description="Topic to search for (e.g., 'machine_learning')"),
    limit: int = Query(default=10, ge=1, le=50, description="Max nodes to return"),
    include_neighbors: bool = Query(default=False, description="Include 1-hop neighbors"),
    graph_store=Depends(get_graph_store)
):
    """Find nodes by topic and optionally include neighbors."""
    try:
        # Query Neo4j for nodes with this topic
        query = """
        MATCH (n:Node)
        WHERE n.topic = $topic
        RETURN n
        LIMIT $limit
        """
        
        result = graph_store._execute_query(query, {"topic": topic, "limit": limit})
        
        nodes = []
        for record in result:
            node_data = dict(record["n"])
            node = graph_store._record_to_node(node_data)
            nodes.append({
                "node": {
                    "id": node.id,
                    "text": node.text,
                    "topic": node.metadata.get("topic"),
                    "category": node.metadata.get("category"),
                    "metadata": node.metadata
                },
                "neighbors": []
            })
            
            # Get neighbors if requested
            if include_neighbors:
                neighbor_query = """
                MATCH (n:Node {id: $node_id})-[r]-(m:Node)
                RETURN m, type(r) as rel_type
                LIMIT 5
                """
                neighbor_result = graph_store._execute_query(neighbor_query, {"node_id": node.id})
                for nr in neighbor_result:
                    neighbor_node = dict(nr["m"])
                    nodes[-1]["neighbors"].append({
                        "id": neighbor_node.get("id"),
                        "topic": neighbor_node.get("topic"),
                        "relationship": nr["rel_type"]
                    })
        
        return {
            "topic": topic,
            "nodes_found": len(nodes),
            "results": nodes
        }
        
    except Exception as e:
        logger.error(f"Topic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/graph/central-nodes",
    summary="Find most connected nodes",
    description="""
    Find the most connected (central) nodes in the graph.
    These are good starting points for exploration.
    Ranking is based on degree (number of connections).
    """
)
async def get_central_nodes(
    limit: int = Query(default=10, ge=1, le=50, description="Number of nodes to return"),
    topic: Optional[str] = Query(default=None, description="Filter by topic"),
    graph_store=Depends(get_graph_store)
):
    """Get the most connected nodes in the graph."""
    try:
        # Query for nodes with highest degree
        if topic:
            query = """
            MATCH (n:Node)
            WHERE n.topic = $topic
            WITH n, size((n)--()) as degree
            ORDER BY degree DESC
            LIMIT $limit
            RETURN n, degree
            """
            params = {"topic": topic, "limit": limit}
        else:
            query = """
            MATCH (n:Node)
            WITH n, size((n)--()) as degree
            ORDER BY degree DESC
            LIMIT $limit
            RETURN n, degree
            """
            params = {"limit": limit}
        
        result = graph_store._execute_query(query, params)
        
        nodes = []
        for record in result:
            node_data = dict(record["n"])
            nodes.append({
                "id": node_data.get("id"),
                "text": node_data.get("text", "")[:200] + "..." if len(node_data.get("text", "")) > 200 else node_data.get("text", ""),
                "topic": node_data.get("topic"),
                "category": node_data.get("category"),
                "degree": record["degree"],
                "metadata": {
                    k: v for k, v in node_data.items() 
                    if k not in ["id", "text", "topic", "category"]
                }
            })
        
        return {
            "description": "Most connected nodes in the graph",
            "filter_topic": topic,
            "results": nodes
        }
        
    except Exception as e:
        logger.error(f"Central nodes search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/graph/search",
    summary="Graph-only search (NO vectors)",
    description="""
    Perform graph-only search using Neo4j without any vector/embedding involvement.
    
    **How it works:**
    1. Filter nodes by topic (optional)
    2. Find nodes whose text contains the query_text (keyword match)
    3. Use matching nodes as starting points for graph traversal
    4. Explore up to `depth` hops from each matched node
    5. Rank results by graph metrics (degree, path weight)
    
    This is PURE GRAPH search - no FAISS, no embeddings.
    """
)
async def graph_only_search(
    query_text: str = Query(default="", description="Keyword to search for in node text"),
    topic: Optional[str] = Query(default=None, description="Filter by topic"),
    depth: int = Query(default=2, ge=1, le=4, description="Traversal depth from matched nodes"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results to return"),
    graph_store=Depends(get_graph_store)
):
    """
    Graph-only search: Find nodes by keyword/topic, then traverse their neighborhoods.
    """
    try:
        matched_nodes = []
        matched_ids = set()
        traversed_nodes = []
        
        # Use the graph_store's session context manager
        with graph_store._session() as session:
            # Step 1: Find starting nodes (by topic and/or keyword)
            # Using COUNT{} instead of size() for Neo4j 5.x compatibility
            if topic and query_text:
                match_query = """
                MATCH (n:Node)
                WHERE n.topic = $topic AND toLower(n.text) CONTAINS toLower($query_text)
                RETURN n, 0 as hop_distance, COUNT { (n)--() } as degree
                """
                params = {"topic": topic, "query_text": query_text}
            elif topic:
                match_query = """
                MATCH (n:Node)
                WHERE n.topic = $topic
                RETURN n, 0 as hop_distance, COUNT { (n)--() } as degree
                """
                params = {"topic": topic}
            elif query_text:
                match_query = """
                MATCH (n:Node)
                WHERE toLower(n.text) CONTAINS toLower($query_text)
                RETURN n, 0 as hop_distance, COUNT { (n)--() } as degree
                """
                params = {"query_text": query_text}
            else:
                match_query = """
                MATCH (n:Node)
                WITH n, COUNT { (n)--() } as degree
                ORDER BY degree DESC
                LIMIT 20
                RETURN n, 0 as hop_distance, degree
                """
                params = {}
            
            # Execute first query
            result = session.run(match_query, params)
            for record in result:
                node_data = dict(record["n"])
                node_id = node_data.get("id")
                if node_id and node_id not in matched_ids:
                    matched_ids.add(node_id)
                    matched_nodes.append({
                        "id": node_id,
                        "text": node_data.get("text", ""),
                        "topic": node_data.get("topic"),
                        "hop_distance": 0,
                        "degree": record["degree"],
                        "is_direct_match": True
                    })
            
            # Step 2: Traverse from matched nodes if depth > 1
            if depth > 1 and matched_nodes:
                matched_id_list = list(matched_ids)[:10]
                
                traverse_query = f"""
                MATCH (start:Node)
                WHERE start.id IN $start_ids
                MATCH path = (start)-[:RELATION*1..{depth - 1}]-(neighbor:Node)
                WHERE NOT neighbor.id IN $start_ids
                WITH neighbor, 
                     min(length(path)) as hop_distance,
                     COUNT {{ (neighbor)--() }} as degree
                RETURN DISTINCT neighbor, hop_distance, degree
                ORDER BY hop_distance, degree DESC
                """
                
                result = session.run(traverse_query, {"start_ids": matched_id_list})
                seen_ids = set(matched_ids)
                for record in result:
                    neighbor_data = dict(record["neighbor"])
                    neighbor_id = neighbor_data.get("id")
                    if neighbor_id and neighbor_id not in seen_ids:
                        seen_ids.add(neighbor_id)
                        traversed_nodes.append({
                            "id": neighbor_id,
                            "text": neighbor_data.get("text", ""),
                            "topic": neighbor_data.get("topic"),
                            "hop_distance": record["hop_distance"],
                            "degree": record["degree"],
                            "is_direct_match": False
                        })
        
        # Step 3: Combine and rank
        all_nodes = matched_nodes + traversed_nodes
        
        # Compute graph-based scores
        max_degree = max((n["degree"] for n in all_nodes), default=1) or 1
        
        for node in all_nodes:
            match_bonus = 1.0 if node["is_direct_match"] else 0.0
            degree_score = node["degree"] / max_degree
            hop_penalty = 1.0 / (1 + node["hop_distance"])
            node["graph_score"] = round(0.4 * match_bonus + 0.3 * degree_score + 0.3 * hop_penalty, 4)
            
            text = node["text"]
            node["text_snippet"] = text[:200] + "..." if len(text) > 200 else text
            del node["text"]
        
        all_nodes.sort(key=lambda x: (-x["graph_score"], x["hop_distance"]))
        results = all_nodes[:top_k]
        
        return {
            "search_type": "graph_only",
            "query_text": query_text or None,
            "topic_filter": topic,
            "depth": depth,
            "direct_matches": len(matched_nodes),
            "traversed_found": len(traversed_nodes),
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Graph-only search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/graph/keyword",
    summary="Find nodes containing keywords (graph-only)",
    description="""
    Search for nodes containing specific keywords in their text.
    This is a TEXT search in Neo4j (not semantic/vector search).
    Use this to find starting points for graph traversal.
    """
)
async def graph_keyword_search(
    keyword: str = Query(..., description="Keyword to search for"),
    limit: int = Query(default=10, ge=1, le=50, description="Max nodes to return"),
    include_neighbors: bool = Query(default=True, description="Include neighbor count"),
    graph_store=Depends(get_graph_store)
):
    """Search nodes by keyword in their text (case-insensitive)."""
    try:
        # Use CONTAINS for keyword search (case-insensitive)
        query = """
        MATCH (n:Node)
        WHERE toLower(n.text) CONTAINS toLower($keyword)
        WITH n, size((n)--()) as degree
        ORDER BY degree DESC
        LIMIT $limit
        RETURN n, degree
        """
        
        result = graph_store._execute_query(query, {"keyword": keyword, "limit": limit})
        
        nodes = []
        for record in result:
            node_data = dict(record["n"])
            text = node_data.get("text", "")
            
            # Highlight keyword in text
            text_lower = text.lower()
            keyword_lower = keyword.lower()
            start_idx = text_lower.find(keyword_lower)
            
            if start_idx != -1:
                # Extract snippet around keyword
                snippet_start = max(0, start_idx - 50)
                snippet_end = min(len(text), start_idx + len(keyword) + 100)
                snippet = ("..." if snippet_start > 0 else "") + text[snippet_start:snippet_end] + ("..." if snippet_end < len(text) else "")
            else:
                snippet = text[:150] + "..." if len(text) > 150 else text
            
            nodes.append({
                "id": node_data.get("id"),
                "text_snippet": snippet,
                "topic": node_data.get("topic"),
                "category": node_data.get("category"),
                "degree": record["degree"] if include_neighbors else None
            })
        
        return {
            "keyword": keyword,
            "nodes_found": len(nodes),
            "note": "This is keyword search (exact match), not semantic search. Use /search/vector for semantic similarity.",
            "results": nodes
        }
        
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/graph/topics",
    summary="List all topics in the graph",
    description="Get a list of all unique topics and their node counts. Useful for browsing."
)
async def list_graph_topics(
    graph_store=Depends(get_graph_store)
):
    """Get all unique topics with counts."""
    try:
        query = """
        MATCH (n:Node)
        WHERE n.topic IS NOT NULL
        RETURN n.topic as topic, count(*) as count
        ORDER BY count DESC
        """
        
        result = graph_store._execute_query(query, {})
        
        topics = [{"topic": record["topic"], "count": record["count"]} for record in result]
        
        return {
            "total_topics": len(topics),
            "topics": topics
        }
        
    except Exception as e:
        logger.error(f"List topics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list topics: {str(e)}")


@router.post(
    "/hybrid",
    summary="Hybrid search (vector + graph) with configurable weights",
    description="""
    Perform hybrid search combining vector similarity and graph signals.
    
    **WEIGHT MODES:**
    1. **Adaptive (default)** - If no weights provided, backend auto-detects:
       - Relationship queries ("between", "connected", etc.) → graph_weight=0.6, vector_weight=0.4
       - Standard queries → vector_weight=0.7, graph_weight=0.3
    
    2. **Explicit** - Provide `vector_weight` and `graph_weight` (must sum to 1.0)
       - `vector_weight=1.0, graph_weight=0.0` → Pure vector search ordering
       - `vector_weight=0.0, graph_weight=1.0` → Pure graph proximity ordering
    
    **How it works:**
    1. Embed query, search FAISS for top candidate_k nodes
    2. For each candidate, compute graph score from Neo4j (degree, edge weights, topic connections)
    3. Normalize cosine & graph scores to 0-1 range
    4. Compute final_score = vector_weight × norm_cosine + graph_weight × norm_graph
    5. Sort by final_score, return top_k results
    
    **Request:**
    ```json
    {
      "query_text": "AI in healthcare",
      "top_k": 10,
      "vector_weight": 0.7,
      "graph_weight": 0.3
    }
    ```
    
    **Response:** Array of results with full scoring breakdown {vector_score, graph_score, final_score}.
    """
)
async def hybrid_search(
    request: HybridSearchRequest,
    hybrid_engine=Depends(get_hybrid_engine),
    graph_store=Depends(get_graph_store)
):
    """
    Perform hybrid search with adaptive or explicit weights.
    
    Args:
        request: HybridSearchRequest with query_text, top_k, candidate_k, and optional weights
        
    Returns:
        List of results with id, text, topic, metadata, cosine_sim, graph_score, final_score
    """
    try:
        # Determine if using adaptive or explicit weights
        use_explicit = request.vector_weight is not None and request.graph_weight is not None
        
        if use_explicit:
            # Validate weights sum to ~1.0
            weight_sum = request.vector_weight + request.graph_weight
            if abs(weight_sum - 1.0) > 0.01:
                raise HTTPException(
                    status_code=400, 
                    detail=f"vector_weight + graph_weight must equal 1.0, got {weight_sum}"
                )
            vector_weight = request.vector_weight
            graph_weight = request.graph_weight
        else:
            # Use adaptive weights (None triggers auto-detection in engine)
            vector_weight = None
            graph_weight = None
        
        response = hybrid_engine.hybrid_search(
            query_text=request.query_text,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            vector_weight=vector_weight,
            graph_weight=graph_weight
        )
        
        # Transform to the required JSON format
        results = []
        for r in response.results:
            # Get additional metadata
            node_metadata = r.node.metadata if r.node.metadata else {}
            topic = node_metadata.get("topic") or r.node.topic
            
            results.append({
                "id": r.node.id,
                "text": r.node.text,
                "topic": topic,
                "metadata": node_metadata,
                "cosine_sim": round(r.cosine_similarity, 4),
                "graph_score": round(r.graph_normalized, 4),  # Use normalized for comparability
                "final_score": round(r.final_score, 4),
                "rank": r.rank,
                "vector_only_rank": r.vector_only_rank,
                "degree": r.degree
            })
        
        return {
            "query_text": request.query_text,
            "total_results": len(results),
            "vector_weight": response.vector_weight_used,
            "graph_weight": response.graph_weight_used,
            "weights_adaptive": response.weights_adaptive,
            "search_time_ms": round(response.search_time_ms, 2),
            "ranking_changed": response.ranking_changed,
            "results": results
        }
        
    except HTTPException:
        raise
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
