"""
Ingestion API Router

Provides endpoints for adding new nodes to the hybrid database.
Handles:
- Single node ingestion with automatic embedding and graph addition
- Automatic relationship creation based on semantic similarity
- Bulk node ingestion
- Automatic snapshot.json and FAISS index updates
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.graph import NodeCreate, EdgeCreate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


# =============================================================================
# Request/Response Models
# =============================================================================

class NodeIngestRequest(BaseModel):
    """Request model for ingesting a single node."""
    text: str = Field(..., description="The text content of the node", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (title, topic, category, source, etc.)"
    )


class BulkIngestRequest(BaseModel):
    """Request model for bulk node ingestion."""
    nodes: List[NodeIngestRequest] = Field(..., description="List of nodes to ingest")


class NodeIngestResponse(BaseModel):
    """Response model for single node ingestion."""
    node: Dict[str, Any]
    edges_created: int = 0
    message: str


class BulkIngestResponse(BaseModel):
    """Response model for bulk ingestion."""
    nodes_created: int
    edges_created: int = 0
    message: str


# =============================================================================
# Helper Functions
# =============================================================================

def get_services():
    """Get the global service instances from main module."""
    from app.main import embedding_service, vector_store, graph_store, snapshot_manager
    
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not available")
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available")
    if not graph_store:
        raise HTTPException(status_code=503, detail="Graph store not available")
    if not snapshot_manager:
        raise HTTPException(status_code=503, detail="Snapshot manager not available")
    
    return embedding_service, vector_store, graph_store, snapshot_manager


def save_faiss_index(vector_store):
    """Save FAISS index to disk."""
    faiss_path = "data/vector_index"
    try:
        vector_store.save(faiss_path)
        logger.info(f"FAISS index saved to {faiss_path}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")


def update_snapshot_with_node(node_id: str, node_data: Dict[str, Any], snapshot_path: Path = Path("data/snapshot.json")):
    """
    Add or update a node in the snapshot.json file.
    
    The snapshot format is:
    {
        "nodes": {
            "doc_001": {"text": "...", "topic": "...", "metadata": {...}},
            ...
        },
        "edges": [...]
    }
    """
    try:
        # Load existing snapshot
        if snapshot_path.exists():
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
        else:
            snapshot = {"nodes": {}, "edges": []}
        
        # Ensure correct format
        if "nodes" not in snapshot or isinstance(snapshot.get("nodes"), list):
            # Convert old format if needed
            old_nodes = snapshot.get("nodes", [])
            snapshot = {"nodes": {}, "edges": snapshot.get("edges", [])}
            if isinstance(old_nodes, list):
                for n in old_nodes:
                    if isinstance(n, dict) and "id" in n:
                        snapshot["nodes"][n["id"]] = n
        
        # Add the new node (dict format with node_id as key)
        snapshot["nodes"][node_id] = node_data
        
        # Save back
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        # Invalidate snapshot manager cache so stats are up to date
        try:
            from app.main import snapshot_manager
            if snapshot_manager:
                snapshot_manager.load_snapshot()
        except Exception:
            pass  # Non-critical if cache refresh fails
        
        logger.info(f"Node {node_id} saved to snapshot.json")
        return True
    except Exception as e:
        logger.error(f"Failed to update snapshot: {e}")
        return False


def update_snapshot_with_edge(edge_data: Dict[str, Any], snapshot_path: Path = Path("data/snapshot.json")):
    """Add an edge to the snapshot.json file."""
    try:
        if snapshot_path.exists():
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
        else:
            snapshot = {"nodes": {}, "edges": []}
        
        if "edges" not in snapshot:
            snapshot["edges"] = []
        
        # Check if edge already exists
        existing_ids = {e.get("id") for e in snapshot["edges"] if isinstance(e, dict)}
        if edge_data.get("id") not in existing_ids:
            snapshot["edges"].append(edge_data)
            
            with open(snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            
            # Invalidate snapshot manager cache so stats are up to date
            try:
                from app.main import snapshot_manager
                if snapshot_manager:
                    snapshot_manager.load_snapshot()
            except Exception:
                pass  # Non-critical if cache refresh fails
            
            logger.info(f"Edge {edge_data.get('id')} saved to snapshot.json")
        return True
    except Exception as e:
        logger.error(f"Failed to update snapshot with edge: {e}")
        return False


def find_and_create_relationships(
    node_id: str,
    node_text: str,
    node_topic: str,
    embedding,
    vector_store,
    graph_store,
    similarity_threshold: float = 0.5,
    max_edges: int = 3
) -> int:
    """
    Find similar nodes and create relationships.
    
    Returns the number of edges created.
    """
    edges_created = 0
    
    try:
        # Search for similar nodes (excluding self)
        similar_nodes = vector_store.search(embedding, top_k=max_edges + 1)
        
        for similar_id, similarity in similar_nodes:
            # Skip self
            if similar_id == node_id:
                continue
            
            # Only create edge if similarity is above threshold
            if similarity < similarity_threshold:
                continue
            
            # Determine relationship type based on similarity
            if similarity > 0.8:
                rel_type = "SIMILAR_TO"
            elif similarity > 0.6:
                rel_type = "RELATED_TO"
            else:
                rel_type = "MENTIONS"
            
            try:
                # Create edge in Neo4j
                edge = graph_store.create_edge(EdgeCreate(
                    source_id=node_id,
                    target_id=similar_id,
                    type=rel_type,
                    weight=round(similarity, 4)
                ))
                
                # Also save to snapshot
                update_snapshot_with_edge({
                    "id": edge.id,
                    "source_id": node_id,
                    "target_id": similar_id,
                    "type": rel_type,
                    "weight": round(similarity, 4)
                })
                
                edges_created += 1
                logger.info(f"Created edge: {node_id} --[{rel_type}:{similarity:.3f}]--> {similar_id}")
                
                if edges_created >= max_edges:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to create edge to {similar_id}: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error finding relationships: {e}")
    
    return edges_created


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/node", response_model=NodeIngestResponse)
async def ingest_single_node(request: NodeIngestRequest):
    """
    Ingest a single node into the hybrid database.
    
    This endpoint:
    1. Generates embedding for the text
    2. Adds the node to Neo4j graph
    3. Adds the embedding to FAISS vector index
    4. Finds similar nodes and creates relationships
    5. Saves to snapshot.json for persistence
    
    Args:
        request: NodeIngestRequest with text and optional metadata
        
    Returns:
        NodeIngestResponse with the created node, edges created, and confirmation message
    """
    embedding_service, vector_store, graph_store, snapshot_manager = get_services()
    
    # Build metadata
    metadata = request.metadata or {}
    title = metadata.get("title") or request.text[:50] + ("..." if len(request.text) > 50 else "")
    topic = metadata.get("topic", "general")
    category = metadata.get("category", "user_input")
    source = metadata.get("source", "web_ingestion")
    
    # Full metadata for storage
    full_metadata = {
        "title": title,
        "category": category,
        "source": source,
        **{k: v for k, v in metadata.items() if k not in ["title", "topic", "category", "source"]}
    }
    
    try:
        # 1. Generate embedding FIRST (before adding to Neo4j, so we can find similar nodes)
        logger.info(f"Generating embedding for new node")
        embedding = embedding_service.embed(request.text)
        
        # 2. Create node in Neo4j using NodeCreate model
        logger.info(f"Adding node to Neo4j")
        node = graph_store.create_node(NodeCreate(
            text=request.text,
            metadata={
                "title": title,
                "topic": topic,
                "category": category,
                "source": source,
            }
        ))
        node_id = node.id
        logger.info(f"Created node {node_id} in Neo4j")
        
        # 3. Add embedding to FAISS vector index
        logger.info(f"Adding node {node_id} to FAISS index")
        vector_store.add_embedding(node_id, embedding)
        
        # 4. Find similar nodes and create relationships
        logger.info(f"Finding relationships for node {node_id}")
        edges_created = find_and_create_relationships(
            node_id=node_id,
            node_text=request.text,
            node_topic=topic,
            embedding=embedding,
            vector_store=vector_store,
            graph_store=graph_store,
            similarity_threshold=0.4,
            max_edges=3
        )
        
        # 5. Save FAISS index to disk
        save_faiss_index(vector_store)
        
        # 6. Save to snapshot.json (dict format)
        snapshot_node_data = {
            "text": request.text,
            "topic": topic,
            "metadata": full_metadata
        }
        update_snapshot_with_node(node_id, snapshot_node_data)
        
        # Build response
        response_node = {
            "id": node_id,
            "text": request.text,
            "topic": topic,
            "metadata": full_metadata,
            "created_at": node.created_at.isoformat() if node.created_at else datetime.utcnow().isoformat()
        }
        
        logger.info(f"Successfully ingested node {node_id} with {edges_created} edges")
        
        return NodeIngestResponse(
            node=response_node,
            edges_created=edges_created,
            message=f"Node '{title}' ingested with ID {node_id}. Created {edges_created} relationship(s)."
        )
        
    except Exception as e:
        logger.error(f"Failed to ingest node: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest node: {str(e)}"
        )


@router.post("/bulk", response_model=BulkIngestResponse)
async def ingest_bulk_nodes(request: BulkIngestRequest):
    """
    Ingest multiple nodes at once.
    
    More efficient than calling /ingest/node multiple times as it batches
    the FAISS save operation. Also creates relationships between ingested nodes.
    
    Args:
        request: BulkIngestRequest with list of nodes
        
    Returns:
        BulkIngestResponse with counts and confirmation message
    """
    embedding_service, vector_store, graph_store, snapshot_manager = get_services()
    
    if not request.nodes:
        raise HTTPException(status_code=400, detail="No nodes provided")
    
    created_count = 0
    total_edges_created = 0
    errors = []
    created_node_ids = []
    
    for i, node_request in enumerate(request.nodes):
        try:
            # Build metadata
            metadata = node_request.metadata or {}
            title = metadata.get("title") or node_request.text[:50] + ("..." if len(node_request.text) > 50 else "")
            topic = metadata.get("topic", "general")
            category = metadata.get("category", "user_input")
            source = metadata.get("source", "bulk_ingestion")
            
            full_metadata = {
                "title": title,
                "category": category,
                "source": source,
            }
            
            # Generate embedding
            embedding = embedding_service.embed(node_request.text)
            
            # Create node in Neo4j
            node = graph_store.create_node(NodeCreate(
                text=node_request.text,
                metadata={
                    "title": title,
                    "topic": topic,
                    "category": category,
                    "source": source,
                }
            ))
            node_id = node.id
            
            # Add to FAISS
            vector_store.add_embedding(node_id, embedding)
            
            # Save to snapshot
            snapshot_node_data = {
                "text": node_request.text,
                "topic": topic,
                "metadata": full_metadata
            }
            update_snapshot_with_node(node_id, snapshot_node_data)
            
            created_count += 1
            created_node_ids.append((node_id, embedding, topic))
            
        except Exception as e:
            logger.error(f"Failed to ingest node {i}: {e}")
            errors.append(f"Node {i}: {str(e)}")
    
    # Create relationships after all nodes are added
    for node_id, embedding, topic in created_node_ids:
        try:
            edges = find_and_create_relationships(
                node_id=node_id,
                node_text="",  # Not needed for similarity search
                node_topic=topic,
                embedding=embedding,
                vector_store=vector_store,
                graph_store=graph_store,
                similarity_threshold=0.4,
                max_edges=2  # Fewer edges per node in bulk
            )
            total_edges_created += edges
        except Exception as e:
            logger.warning(f"Failed to create relationships for {node_id}: {e}")
    
    # Save FAISS index once at the end
    if created_count > 0:
        save_faiss_index(vector_store)
    
    message = f"Successfully ingested {created_count} of {len(request.nodes)} nodes with {total_edges_created} relationships"
    if errors:
        message += f". Errors: {', '.join(errors[:3])}"
        if len(errors) > 3:
            message += f" and {len(errors) - 3} more"
    
    return BulkIngestResponse(
        nodes_created=created_count,
        edges_created=total_edges_created,
        message=message
    )
