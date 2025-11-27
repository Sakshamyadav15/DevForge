"""
Edge API Routes

This module provides REST API endpoints for Edge/Relationship CRUD operations:
- POST /edges - Create a new edge between nodes
- GET /edges - List all edges
- GET /edges/{id} - Get a specific edge
- DELETE /edges/{id} - Delete an edge

Edges represent relationships between nodes in the knowledge graph.
They have a type (e.g., RELATED_TO, CITES) and a weight (0.0-1.0).
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from app.models.graph import Edge, EdgeCreate
from app.models.search import PaginatedResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/edges", tags=["Edges"])


# Dependency injection for services
def get_graph_store():
    """Dependency to get GraphStore instance."""
    from app.main import graph_store
    return graph_store


def get_snapshot_manager():
    """Dependency to get SnapshotManager instance."""
    from app.main import snapshot_manager
    return snapshot_manager


@router.post(
    "",
    response_model=Edge,
    status_code=201,
    summary="Create a new edge",
    description="""
    Create a relationship between two existing nodes.
    
    The edge has:
    - source_id: The source node ID
    - target_id: The target node ID  
    - type: Relationship type (e.g., RELATED_TO, CITES, SUPPORTS)
    - weight: Strength of relationship (0.0 to 1.0)
    
    If an edge with the same source, target, and type already exists,
    its weight is updated instead of creating a duplicate.
    """
)
async def create_edge(
    edge_data: EdgeCreate,
    graph_store=Depends(get_graph_store),
    snapshot_manager=Depends(get_snapshot_manager)
) -> Edge:
    """
    Create a new edge between two nodes.
    
    Args:
        edge_data: Edge creation data (source_id, target_id, type, weight)
        
    Returns:
        The created Edge with assigned ID
        
    Raises:
        HTTPException 400: If source or target node doesn't exist
        HTTPException 500: If creation fails
    """
    try:
        # Verify both nodes exist
        source_node = graph_store.get_node(edge_data.source_id)
        if not source_node:
            raise HTTPException(
                status_code=400,
                detail=f"Source node '{edge_data.source_id}' not found"
            )
        
        target_node = graph_store.get_node(edge_data.target_id)
        if not target_node:
            raise HTTPException(
                status_code=400,
                detail=f"Target node '{edge_data.target_id}' not found"
            )
        
        # Create edge in Neo4j
        edge = graph_store.create_edge(edge_data)
        
        # Persist to snapshot
        snapshot_manager.append_edge(edge)
        
        logger.info(f"Created edge: {edge.id}")
        return edge
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create edge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create edge: {str(e)}")


@router.get(
    "",
    response_model=PaginatedResponse,
    summary="List all edges",
    description="Retrieve all edges with optional pagination and filtering."
)
async def list_edges(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=10, ge=1, le=100, description="Items per page"),
    limit: Optional[int] = Query(default=None, ge=1, le=100, description="Items per page (alias)"),
    offset: Optional[int] = Query(default=None, ge=0, description="Offset"),
    edge_type: Optional[str] = Query(default=None, alias="type", description="Filter by edge type"),
    source_id: Optional[str] = Query(default=None, alias="sourceId", description="Filter by source node"),
    target_id: Optional[str] = Query(default=None, alias="targetId", description="Filter by target node"),
    graph_store=Depends(get_graph_store),
    snapshot_manager=Depends(get_snapshot_manager)
) -> PaginatedResponse:
    """
    List all edges with pagination and optional filtering.
    """
    try:
        # Use limit/offset if provided
        if limit is not None:
            page_size = limit
        if offset is not None:
            page = (offset // page_size) + 1
        
        all_edges = []
        
        # Try graph store first, fall back to snapshot
        if graph_store is not None:
            try:
                all_edges = graph_store.get_all_edges()
            except Exception as e:
                logger.warning(f"Graph store failed, falling back to snapshot: {e}")
        
        # Fall back to snapshot if Neo4j not available or failed
        if not all_edges and snapshot_manager:
            snapshot_edges = snapshot_manager.get_all_edges()
            for edge_data in snapshot_edges:
                all_edges.append(Edge(
                    id=edge_data.get("id", f"{edge_data.get('source_id')}_{edge_data.get('target_id')}_{edge_data.get('type')}"),
                    source_id=edge_data.get("source_id"),
                    target_id=edge_data.get("target_id"),
                    type=edge_data.get("type", "RELATED_TO"),
                    weight=edge_data.get("weight", 1.0)
                ))
        
        # Apply filters
        if edge_type:
            all_edges = [e for e in all_edges if e.type == edge_type]
        if source_id:
            all_edges = [e for e in all_edges if e.source_id == source_id]
        if target_id:
            all_edges = [e for e in all_edges if e.target_id == target_id]
        
        # Calculate pagination
        total = len(all_edges)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        # Slice for current page
        start = (page - 1) * page_size
        end = start + page_size
        page_items = all_edges[start:end]
        
        # Transform to frontend format
        items = []
        for e in page_items:
            items.append({
                "id": e.id,
                "source": e.source_id,
                "target": e.target_id,
                "type": e.type,
                "weight": e.weight,
                "created_at": e.created_at.isoformat() if e.created_at else None
            })
        
        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except Exception as e:
        logger.error(f"Failed to list edges: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list edges: {str(e)}")


@router.get(
    "/{edge_id}",
    response_model=Edge,
    summary="Get an edge by ID",
    description="Retrieve a specific edge by its ID."
)
async def get_edge(
    edge_id: str,
    graph_store=Depends(get_graph_store)
) -> Edge:
    """
    Get an edge by ID.
    
    Args:
        edge_id: The unique edge identifier
        
    Returns:
        The Edge if found
        
    Raises:
        HTTPException 404: If edge not found
    """
    edge = graph_store.get_edge(edge_id)
    
    if not edge:
        raise HTTPException(status_code=404, detail=f"Edge '{edge_id}' not found")
    
    return edge


@router.get(
    "/between/{source_id}/{target_id}",
    response_model=list[Edge],
    summary="Get edges between two nodes",
    description="Retrieve all edges connecting two specific nodes."
)
async def get_edges_between(
    source_id: str,
    target_id: str,
    graph_store=Depends(get_graph_store)
) -> list[Edge]:
    """
    Get all edges between two nodes.
    
    Args:
        source_id: The source node ID
        target_id: The target node ID
        
    Returns:
        List of edges connecting the two nodes
    """
    try:
        all_edges = graph_store.get_all_edges()
        
        # Filter edges between the two nodes (in either direction)
        edges_between = [
            e for e in all_edges
            if (e.source_id == source_id and e.target_id == target_id) or
               (e.source_id == target_id and e.target_id == source_id)
        ]
        
        return edges_between
        
    except Exception as e:
        logger.error(f"Failed to get edges between nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{edge_id}",
    status_code=204,
    summary="Delete an edge",
    description="Delete a relationship by its ID."
)
async def delete_edge(
    edge_id: str,
    graph_store=Depends(get_graph_store),
    snapshot_manager=Depends(get_snapshot_manager)
) -> None:
    """
    Delete an edge by ID.
    
    Args:
        edge_id: The edge ID to delete
        
    Raises:
        HTTPException 404: If edge not found
        HTTPException 500: If deletion fails
    """
    try:
        # Delete from Neo4j
        deleted = graph_store.delete_edge(edge_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Edge '{edge_id}' not found")
        
        # Delete from snapshot
        snapshot_manager.delete_edge(edge_id)
        
        logger.info(f"Deleted edge: {edge_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete edge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete edge: {str(e)}")
