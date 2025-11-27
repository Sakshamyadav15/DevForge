"""
Graph Data Models

This module defines Pydantic models for graph entities:
- Node: Represents a document/knowledge node with text and metadata
- Edge: Represents relationships between nodes with type and weight

These models are used for:
- Request/response validation in API endpoints
- Data serialization for Neo4j storage
- JSON snapshot persistence
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class NodeBase(BaseModel):
    """
    Base model for Node with common fields.
    
    Attributes:
        text: The main content/text of the node (e.g., document text, note content)
        metadata: Free-form dictionary for additional info (source, topic, tags, etc.)
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The main text content of the node"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (e.g., source, topic, tags)"
    )


class NodeCreate(NodeBase):
    """
    Model for creating a new node.
    
    The ID is auto-generated if not provided.
    
    Example:
        {
            "text": "AI is transforming healthcare through medical imaging",
            "metadata": {"source": "research_paper", "topic": "healthcare"}
        }
    """
    id: Optional[str] = Field(
        default=None,
        description="Optional custom ID; auto-generated if not provided"
    )


class NodeUpdate(BaseModel):
    """
    Model for updating an existing node.
    
    All fields are optional - only provided fields will be updated.
    """
    text: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=50000,
        description="Updated text content"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Updated metadata (replaces existing metadata)"
    )


class Node(NodeBase):
    """
    Complete Node model with all fields.
    
    This is the canonical representation of a node returned by the API.
    The embedding is NOT stored in this model - it exists only in FAISS
    and can be regenerated from the text.
    
    Attributes:
        id: Unique identifier for the node
        text: The main content of the node
        metadata: Free-form metadata dictionary
        created_at: Timestamp when the node was created
        updated_at: Timestamp of last update
    """
    id: str = Field(
        ...,
        description="Unique identifier for the node"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the node was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last update"
    )
    
    class Config:
        """Pydantic model configuration."""
        # Allow creating from ORM objects and dicts
        from_attributes = True
        # Example for OpenAPI documentation
        json_schema_extra = {
            "example": {
                "id": "node_abc123",
                "text": "AI is transforming healthcare through medical imaging and diagnostics",
                "metadata": {
                    "source": "research_paper",
                    "topic": "healthcare",
                    "tags": ["AI", "medical", "imaging"]
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class NodeWithNeighbors(Node):
    """
    Node model extended with neighbor information.
    
    Used when fetching a node with its direct connections.
    """
    neighbors: list["Node"] = Field(
        default_factory=list,
        description="List of directly connected nodes"
    )
    edge_count: int = Field(
        default=0,
        description="Total number of edges connected to this node"
    )


class EdgeBase(BaseModel):
    """
    Base model for Edge with common fields.
    
    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        type: Relationship type (e.g., RELATED_TO, CITES, SUPPORTS)
        weight: Strength of the relationship (0.0 to 1.0)
    """
    source_id: str = Field(
        ...,
        description="ID of the source node"
    )
    target_id: str = Field(
        ...,
        description="ID of the target node"
    )
    type: str = Field(
        default="RELATED_TO",
        description="Relationship type (e.g., RELATED_TO, CITES, SUPPORTS)"
    )
    weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the relationship (0.0 to 1.0)"
    )


class EdgeCreate(EdgeBase):
    """
    Model for creating a new edge/relationship.
    
    Example:
        {
            "source_id": "node_1",
            "target_id": "node_2",
            "type": "CITES",
            "weight": 0.9
        }
    """
    pass


class Edge(EdgeBase):
    """
    Complete Edge model with all fields.
    
    Represents a directed relationship between two nodes in the graph.
    The edge ID is synthetic (generated from source_id, target_id, and type).
    
    Attributes:
        id: Synthetic unique identifier for the edge
        source_id: ID of the source node
        target_id: ID of the target node
        type: Relationship type
        weight: Relationship strength
        created_at: Timestamp when the edge was created
    """
    id: str = Field(
        ...,
        description="Unique identifier for the edge"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the edge was created"
    )
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "edge_node1_node2_CITES",
                "source_id": "node_1",
                "target_id": "node_2",
                "type": "CITES",
                "weight": 0.9,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


def generate_node_id() -> str:
    """
    Generate a unique node ID using UUID4.
    
    Returns:
        A unique string ID prefixed with 'node_'
    """
    return f"node_{uuid.uuid4().hex[:12]}"


def generate_edge_id(source_id: str, target_id: str, edge_type: str) -> str:
    """
    Generate a synthetic edge ID from source, target, and type.
    
    This makes edges identifiable and prevents duplicate edges of the same type
    between the same nodes.
    
    Args:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Type of the relationship
        
    Returns:
        A synthetic edge ID
    """
    return f"edge_{source_id}_{target_id}_{edge_type}"


# Enable forward references for NodeWithNeighbors
NodeWithNeighbors.model_rebuild()
