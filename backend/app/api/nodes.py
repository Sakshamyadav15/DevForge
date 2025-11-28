"""
Node API Routes

This module provides REST API endpoints for Node CRUD operations:
- POST /nodes - Create a new node
- GET /nodes - List all nodes (with pagination)
- GET /nodes/{id} - Get a specific node
- PUT /nodes/{id} - Update a node
- DELETE /nodes/{id} - Delete a node

Each operation maintains consistency across:
- Neo4j (graph structure)
- FAISS (vector embeddings)
- Snapshot (JSON persistence)
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime

from app.models.graph import Node, NodeCreate, NodeUpdate, NodeWithNeighbors
from app.models.search import PaginatedResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nodes", tags=["Nodes"])


# Dependency injection for services (will be set up in main.py)
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


def get_snapshot_manager():
    """Dependency to get SnapshotManager instance."""
    from app.main import snapshot_manager
    return snapshot_manager


@router.post(
    "",
    response_model=Node,
    status_code=201,
    summary="Create a new node",
    description="""
    Create a new knowledge node in the graph.
    
    This operation:
    1. Creates the node in Neo4j with text and metadata
    2. Generates an embedding and adds it to FAISS
    3. Persists the node to the JSON snapshot
    
    If no ID is provided, one is automatically generated.
    """
)
async def create_node(
    node_data: NodeCreate,
    graph_store=Depends(get_graph_store),
    vector_store=Depends(get_vector_store),
    embedding_service=Depends(get_embedding_service),
    snapshot_manager=Depends(get_snapshot_manager)
) -> Node:
    """
    Create a new node in the knowledge graph.
    
    Args:
        node_data: Node creation data (text, metadata, optional id)
        
    Returns:
        The created Node with assigned ID and timestamps
        
    Raises:
        HTTPException 400: If validation fails
        HTTPException 500: If creation fails
    """
    try:
        # Create node in Neo4j
        node = graph_store.create_node(node_data)
        
        # Generate embedding and add to vector store
        embedding = embedding_service.embed(node.text)
        vector_store.add_embedding(node.id, embedding)
        
        # Persist to snapshot
        snapshot_manager.append_node(node)
        
        logger.info(f"Created node: {node.id}")
        return node
        
    except Exception as e:
        logger.error(f"Failed to create node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create node: {str(e)}")


@router.get(
    "",
    response_model=PaginatedResponse,
    summary="List all nodes",
    description="Retrieve all nodes with optional pagination."
)
async def list_nodes(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=10, ge=1, le=500, description="Items per page"),
    limit: Optional[int] = Query(default=None, ge=1, le=500, description="Items per page (alias for page_size)"),
    offset: Optional[int] = Query(default=None, ge=0, description="Offset (alternative to page)"),
    source: Optional[str] = Query(default=None, description="Filter by source metadata"),
    topic: Optional[str] = Query(default=None, description="Filter by topic metadata"),
    category: Optional[str] = Query(default=None, description="Filter by category metadata"),
    graph_store=Depends(get_graph_store),
    snapshot_manager=Depends(get_snapshot_manager)
) -> PaginatedResponse:
    """
    List all nodes with pagination and optional filtering.
    
    Supports both page-based (page/page_size) and offset-based (limit/offset) pagination.
    """
    try:
        # Use limit/offset if provided, otherwise use page/page_size
        if limit is not None:
            page_size = limit
        if offset is not None:
            page = (offset // page_size) + 1
        
        all_nodes = []
        
        # Try graph store first, fall back to snapshot
        if graph_store is not None:
            try:
                all_nodes = graph_store.get_all_nodes()
            except Exception as e:
                logger.warning(f"Graph store failed, falling back to snapshot: {e}")
                graph_store_failed = True
        
        # Fall back to snapshot if Neo4j not available or failed
        if not all_nodes and snapshot_manager:
            snapshot_nodes = snapshot_manager.get_all_nodes()
            for node_id, node_data in snapshot_nodes.items():
                all_nodes.append(Node(
                    id=node_id,
                    text=node_data.get("text", ""),
                    metadata=node_data.get("metadata", {}),
                    created_at=node_data.get("created_at"),
                    updated_at=node_data.get("updated_at")
                ))
        
        # Apply filters
        if source:
            all_nodes = [n for n in all_nodes if n.metadata.get("source") == source]
        if topic:
            all_nodes = [n for n in all_nodes if n.metadata.get("topic") == topic]
        if category:
            all_nodes = [n for n in all_nodes if n.metadata.get("category") == category]
        
        # Calculate pagination
        total = len(all_nodes)
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1
        
        # Slice for current page
        start = (page - 1) * page_size
        end = start + page_size
        page_items = all_nodes[start:end]
        
        # Transform to frontend-compatible format
        items = []
        for n in page_items:
            items.append({
                "id": n.id,
                "title": n.metadata.get("title"),
                "text": n.text[:500] if n.text else None,
                "topic": n.metadata.get("topic"),
                "category": n.metadata.get("category"),
                "created_at": n.created_at.isoformat() if n.created_at else None,
                "metadata": n.metadata
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
        logger.error(f"Failed to list nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list nodes: {str(e)}")


@router.get(
    "/{node_id}",
    response_model=NodeWithNeighbors,
    summary="Get a node by ID",
    description="Retrieve a specific node with its direct neighbors."
)
async def get_node(
    node_id: str,
    include_neighbors: bool = Query(default=True, description="Include direct neighbors"),
    graph_store=Depends(get_graph_store),
    snapshot_manager=Depends(get_snapshot_manager)
) -> NodeWithNeighbors:
    """
    Get a node by ID with optional neighbor information.
    
    Args:
        node_id: The unique node identifier
        include_neighbors: Whether to include directly connected nodes
        
    Returns:
        The node with neighbor information
        
    Raises:
        HTTPException 404: If node not found
    """
    node = None
    neighbors = []
    edge_count = 0
    
    # Try graph store first
    if graph_store is not None:
        try:
            node = graph_store.get_node(node_id)
            if node and include_neighbors:
                edges = graph_store.get_edges_for_node(node_id)
                edge_count = len(edges)
                
                neighbor_ids = set()
                for edge in edges:
                    if edge.source_id == node_id:
                        neighbor_ids.add(edge.target_id)
                    else:
                        neighbor_ids.add(edge.source_id)
                
                for neighbor_id in neighbor_ids:
                    neighbor_node = graph_store.get_node(neighbor_id)
                    if neighbor_node:
                        neighbors.append(neighbor_node)
        except Exception as e:
            logger.warning(f"Graph store failed: {e}")
    
    # Fall back to snapshot if graph store failed or unavailable
    if node is None and snapshot_manager:
        node_data = snapshot_manager.get_node(node_id)
        if node_data:
            node = Node(
                id=node_id,
                text=node_data.get("text", ""),
                metadata=node_data.get("metadata", {}),
                created_at=node_data.get("created_at"),
                updated_at=node_data.get("updated_at")
            )
            # No neighbor info from snapshot in this simple case
    
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    
    return NodeWithNeighbors(
        id=node.id,
        text=node.text,
        metadata=node.metadata,
        created_at=node.created_at,
        updated_at=node.updated_at,
        neighbors=neighbors,
        edge_count=edge_count
    )


@router.get(
    "/{node_id}/neighbors",
    summary="Get neighbors for a node (frontend-compatible)",
    description="Returns nodes and edges connected to this node."
)
async def get_node_neighbors(
    node_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Maximum neighbors to return"),
    graph_store=Depends(get_graph_store)
):
    """
    Get neighbors and edges for a node.
    
    Returns:
        {nodes: Node[], edges: Edge[]} for frontend graph visualization
    """
    node = graph_store.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    
    edges = graph_store.get_edges_for_node(node_id)
    
    neighbor_ids = set()
    for edge in edges:
        if edge.source_id == node_id:
            neighbor_ids.add(edge.target_id)
        else:
            neighbor_ids.add(edge.source_id)
    
    neighbors = []
    for neighbor_id in list(neighbor_ids)[:limit]:
        neighbor_node = graph_store.get_node(neighbor_id)
        if neighbor_node:
            neighbors.append({
                "id": neighbor_node.id,
                "text": neighbor_node.text[:200] if neighbor_node.text else None,
                "topic": neighbor_node.metadata.get("topic"),
                "category": neighbor_node.metadata.get("category"),
                "created_at": neighbor_node.created_at.isoformat() if neighbor_node.created_at else None,
                "metadata": neighbor_node.metadata
            })
    
    edge_list = []
    for edge in edges[:limit]:
        edge_list.append({
            "id": f"{edge.source_id}_{edge.target_id}_{edge.type}",
            "source": edge.source_id,
            "target": edge.target_id,
            "type": edge.type,
            "weight": edge.weight,
            "created_at": edge.created_at.isoformat() if edge.created_at else None
        })
    
    return {"nodes": neighbors, "edges": edge_list}


@router.put(
    "/{node_id}",
    response_model=Node,
    summary="Update a node",
    description="""
    Update an existing node's text and/or metadata.
    
    If the text is updated, the embedding is automatically regenerated.
    You can also force embedding regeneration using regen_embedding=true.
    """
)
async def update_node(
    node_id: str,
    update_data: NodeUpdate,
    regen_embedding: bool = Query(
        default=False, 
        description="Force regenerate embedding even if text unchanged"
    ),
    graph_store=Depends(get_graph_store),
    vector_store=Depends(get_vector_store),
    embedding_service=Depends(get_embedding_service),
    snapshot_manager=Depends(get_snapshot_manager)
) -> Node:
    """
    Update a node's text and/or metadata.
    
    Args:
        node_id: The node ID to update
        update_data: Fields to update
        regen_embedding: Force embedding regeneration (default: False)
        
    Returns:
        The updated node
        
    Raises:
        HTTPException 404: If node not found
        HTTPException 500: If update fails
    """
    try:
        # Get old embedding for comparison if needed
        old_embedding = None
        if regen_embedding or update_data.text is not None:
            old_embedding = vector_store.get_embedding(node_id)
        
        # Update in Neo4j
        updated_node = graph_store.update_node(node_id, update_data)
        
        if not updated_node:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        
        # Regenerate embedding if text was updated OR if explicitly requested
        if update_data.text is not None or regen_embedding:
            new_embedding = embedding_service.embed(updated_node.text)
            vector_store.add_embedding(node_id, new_embedding)  # This updates if exists
            logger.info(f"Regenerated embedding for node: {node_id}")
        
        # Update snapshot
        snapshot_manager.update_node(
            node_id,
            text=update_data.text,
            metadata=update_data.metadata
        )
        
        logger.info(f"Updated node: {node_id}")
        return updated_node
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update node: {str(e)}")


@router.delete(
    "/{node_id}",
    status_code=204,
    summary="Delete a node",
    description="""
    Delete a node and all its relationships.
    
    This removes:
    - The node from Neo4j (with all connected edges)
    - The embedding from FAISS
    - The node and its edges from the snapshot
    """
)
async def delete_node(
    node_id: str,
    graph_store=Depends(get_graph_store),
    vector_store=Depends(get_vector_store),
    snapshot_manager=Depends(get_snapshot_manager)
) -> None:
    """
    Delete a node and all its relationships.
    
    Args:
        node_id: The node ID to delete
        
    Raises:
        HTTPException 404: If node not found
        HTTPException 500: If deletion fails
    """
    try:
        # Delete from Neo4j (DETACH DELETE removes edges too)
        deleted = graph_store.delete_node(node_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        
        # Delete from vector store
        vector_store.delete_embedding(node_id)
        
        # Delete from snapshot (also removes connected edges)
        snapshot_manager.delete_node(node_id)
        
        logger.info(f"Deleted node: {node_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete node: {str(e)}")
