"""
Snapshot Manager - Local JSON Persistence

This module provides local persistence for the knowledge graph using JSON snapshots.
The snapshot file is the SOURCE OF TRUTH - Neo4j and FAISS are indexes/engines
built on top of this snapshot data.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    snapshot.json                             │
    │                   (Ground Truth)                             │
    └───────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │   Neo4j   │   │   FAISS   │   │  In-mem   │
    │  (Graph)  │   │ (Vectors) │   │  (Cache)  │
    └───────────┘   └───────────┘   └───────────┘

On startup:
1. Load snapshot from JSON file
2. Clear and rebuild Neo4j graph
3. Regenerate embeddings and rebuild FAISS index

This approach:
- Ensures consistency between all data stores
- Enables disaster recovery from a single file
- Mimics real database WAL/snapshot recovery patterns
- Works offline (no external dependencies for data)
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from app.config import settings, get_snapshot_path
from app.models.graph import Node, Edge, generate_node_id, generate_edge_id

logger = logging.getLogger(__name__)


@dataclass
class SnapshotData:
    """
    Data structure representing a complete snapshot.
    
    Attributes:
        nodes: Dictionary mapping node_id -> node data
        edges: List of edge data dictionaries
        metadata: Snapshot-level metadata (version, created_at, etc.)
    """
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary for JSON serialization."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotData":
        """Create SnapshotData from a dictionary."""
        return cls(
            nodes=data.get("nodes", {}),
            edges=data.get("edges", []),
            metadata=data.get("metadata", {})
        )


class SnapshotManager:
    """
    Manager for JSON snapshot persistence.
    
    This class handles:
    - Loading and saving snapshots to JSON file
    - Incremental updates (append/delete operations)
    - Rebuilding Neo4j and FAISS indexes from snapshot
    
    The snapshot file structure:
    {
        "nodes": {
            "node_id1": {"text": "...", "metadata": {...}},
            "node_id2": {"text": "...", "metadata": {...}}
        },
        "edges": [
            {"source_id": "node_id1", "target_id": "node_id2", "type": "RELATED_TO", "weight": 0.9}
        ],
        "metadata": {
            "version": "1.0",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:30:00Z"
        }
    }
    
    Example:
        >>> manager = SnapshotManager()
        >>> data = manager.load_snapshot()
        >>> manager.append_node(Node(id="n1", text="Hello", metadata={}))
        >>> manager.save_snapshot_full(data)
    """
    
    def __init__(self, snapshot_path: Optional[str] = None):
        """
        Initialize the snapshot manager.
        
        Args:
            snapshot_path: Path to snapshot file (default: from settings)
        """
        self.snapshot_path = Path(snapshot_path) if snapshot_path else get_snapshot_path()
        
        # Ensure parent directory exists
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of current snapshot
        self._cache: Optional[SnapshotData] = None
        
        logger.info(f"SnapshotManager initialized with path: {self.snapshot_path}")
    
    def load_snapshot(self) -> SnapshotData:
        """
        Load snapshot from JSON file.
        
        If file doesn't exist, returns empty snapshot data.
        Caches the result for subsequent operations.
        
        Returns:
            SnapshotData containing nodes and edges
        """
        if not self.snapshot_path.exists():
            logger.info("Snapshot file not found, starting with empty data")
            self._cache = SnapshotData(
                metadata={
                    "version": "1.0",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            return self._cache
        
        try:
            with open(self.snapshot_path, 'r', encoding='utf-8-sig') as f:
                content = f.read().strip()
            
            # Handle empty or whitespace-only files
            if not content:
                logger.warning("Snapshot file is empty, starting with empty data")
                self._cache = SnapshotData(
                    metadata={
                        "version": "1.0",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                )
                # Write valid empty JSON to fix the file
                self.save_snapshot()
                return self._cache
            
            data = json.loads(content)
            
            # Handle both formats:
            # 1. List format: [{"id": "...", "text": "...", ...}, ...]
            # 2. Dict format: {"nodes": {...}, "edges": [...], "metadata": {...}}
            if isinstance(data, list):
                # Convert list format to dict format
                logger.info(f"Converting list format snapshot ({len(data)} items) to dict format...")
                nodes_dict = {}
                for item in data:
                    node_id = item.get("id")
                    if node_id:
                        nodes_dict[node_id] = {
                            "text": item.get("text", item.get("title", "")),
                            "metadata": item.get("metadata", {}),
                            "created_at": item.get("created_at", datetime.utcnow().isoformat()),
                            "updated_at": item.get("updated_at", datetime.utcnow().isoformat())
                        }
                        # Include title in metadata if present
                        if "title" in item and "title" not in nodes_dict[node_id]["metadata"]:
                            nodes_dict[node_id]["metadata"]["title"] = item["title"]
                
                self._cache = SnapshotData(
                    nodes=nodes_dict,
                    edges=[],
                    metadata={
                        "version": "1.0",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "node_count": len(nodes_dict),
                        "edge_count": 0
                    }
                )
            else:
                self._cache = SnapshotData.from_dict(data)
            
            logger.info(
                f"Loaded snapshot: {len(self._cache.nodes)} nodes, "
                f"{len(self._cache.edges)} edges"
            )
            return self._cache
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse snapshot JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            raise
    
    def save_snapshot(
        self,
        nodes: dict[str, Node],
        edges: list[Edge]
    ) -> None:
        """
        Save a complete snapshot to JSON file.
        
        This overwrites the existing snapshot file with new data.
        
        Args:
            nodes: Dictionary of node_id -> Node objects
            edges: List of Edge objects
        """
        # Convert models to serializable dictionaries
        nodes_dict = {}
        for node_id, node in nodes.items():
            nodes_dict[node_id] = {
                "text": node.text,
                "metadata": node.metadata,
                "created_at": node.created_at.isoformat() if node.created_at else None,
                "updated_at": node.updated_at.isoformat() if node.updated_at else None
            }
        
        edges_list = []
        for edge in edges:
            edges_list.append({
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "type": edge.type,
                "weight": edge.weight,
                "created_at": edge.created_at.isoformat() if edge.created_at else None
            })
        
        # Create snapshot data
        snapshot = SnapshotData(
            nodes=nodes_dict,
            edges=edges_list,
            metadata={
                "version": "1.0",
                "created_at": self._cache.metadata.get("created_at") if self._cache else datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "node_count": len(nodes_dict),
                "edge_count": len(edges_list)
            }
        )
        
        # Write to file
        self._write_snapshot(snapshot)
        self._cache = snapshot
    
    def save_current_snapshot(self) -> None:
        """
        Save the current cache state to the snapshot file.
        
        This is useful after making incremental changes via append_node/append_edge.
        """
        if self._cache is None:
            logger.warning("No cache loaded, nothing to save")
            return
        
        # Update metadata
        self._cache.metadata["updated_at"] = datetime.utcnow().isoformat()
        self._cache.metadata["node_count"] = len(self._cache.nodes)
        self._cache.metadata["edge_count"] = len(self._cache.edges)
        
        # Write to file
        self._write_snapshot(self._cache)
        logger.debug("Saved current snapshot to file")
    
    def append_node(self, node: Node) -> None:
        """
        Add a node to the snapshot (incremental update).
        
        This is more efficient than saving the entire snapshot for single additions.
        
        Args:
            node: The Node to add
        """
        # Ensure cache is loaded
        if self._cache is None:
            self.load_snapshot()
        
        # Add node to cache
        self._cache.nodes[node.id] = {
            "text": node.text,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat() if node.created_at else datetime.utcnow().isoformat(),
            "updated_at": node.updated_at.isoformat() if node.updated_at else datetime.utcnow().isoformat()
        }
        
        # Update metadata
        self._cache.metadata["updated_at"] = datetime.utcnow().isoformat()
        self._cache.metadata["node_count"] = len(self._cache.nodes)
        
        # Persist
        self._write_snapshot(self._cache)
        logger.debug(f"Appended node to snapshot: {node.id}")
    
    def update_node(self, node_id: str, text: Optional[str] = None, metadata: Optional[dict] = None) -> bool:
        """
        Update an existing node in the snapshot.
        
        Args:
            node_id: ID of the node to update
            text: New text (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if node was updated, False if not found
        """
        if self._cache is None:
            self.load_snapshot()
        
        if node_id not in self._cache.nodes:
            return False
        
        if text is not None:
            self._cache.nodes[node_id]["text"] = text
        
        if metadata is not None:
            self._cache.nodes[node_id]["metadata"] = metadata
        
        self._cache.nodes[node_id]["updated_at"] = datetime.utcnow().isoformat()
        self._cache.metadata["updated_at"] = datetime.utcnow().isoformat()
        
        self._write_snapshot(self._cache)
        logger.debug(f"Updated node in snapshot: {node_id}")
        return True
    
    def delete_node(self, node_id: str) -> bool:
        """
        Remove a node from the snapshot.
        
        Also removes all edges connected to this node.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            True if node was deleted, False if not found
        """
        if self._cache is None:
            self.load_snapshot()
        
        if node_id not in self._cache.nodes:
            return False
        
        # Remove node
        del self._cache.nodes[node_id]
        
        # Remove all edges involving this node
        self._cache.edges = [
            e for e in self._cache.edges
            if e["source_id"] != node_id and e["target_id"] != node_id
        ]
        
        # Update metadata
        self._cache.metadata["updated_at"] = datetime.utcnow().isoformat()
        self._cache.metadata["node_count"] = len(self._cache.nodes)
        self._cache.metadata["edge_count"] = len(self._cache.edges)
        
        # Persist
        self._write_snapshot(self._cache)
        logger.debug(f"Deleted node from snapshot: {node_id}")
        return True
    
    def append_edge(self, edge: Edge) -> None:
        """
        Add an edge to the snapshot.
        
        Args:
            edge: The Edge to add
        """
        if self._cache is None:
            self.load_snapshot()
        
        # Check for existing edge with same source, target, type
        for existing in self._cache.edges:
            if (existing["source_id"] == edge.source_id and
                existing["target_id"] == edge.target_id and
                existing["type"] == edge.type):
                # Update existing edge
                existing["weight"] = edge.weight
                existing["id"] = edge.id
                self._cache.metadata["updated_at"] = datetime.utcnow().isoformat()
                self._write_snapshot(self._cache)
                logger.debug(f"Updated existing edge in snapshot: {edge.id}")
                return
        
        # Add new edge
        self._cache.edges.append({
            "id": edge.id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "type": edge.type,
            "weight": edge.weight,
            "created_at": edge.created_at.isoformat() if edge.created_at else datetime.utcnow().isoformat()
        })
        
        # Update metadata
        self._cache.metadata["updated_at"] = datetime.utcnow().isoformat()
        self._cache.metadata["edge_count"] = len(self._cache.edges)
        
        # Persist
        self._write_snapshot(self._cache)
        logger.debug(f"Appended edge to snapshot: {edge.id}")
    
    def delete_edge(self, edge_id: str) -> bool:
        """
        Remove an edge from the snapshot.
        
        Args:
            edge_id: ID of the edge to delete
            
        Returns:
            True if edge was deleted, False if not found
        """
        if self._cache is None:
            self.load_snapshot()
        
        original_count = len(self._cache.edges)
        self._cache.edges = [e for e in self._cache.edges if e.get("id") != edge_id]
        
        if len(self._cache.edges) == original_count:
            return False
        
        # Update metadata
        self._cache.metadata["updated_at"] = datetime.utcnow().isoformat()
        self._cache.metadata["edge_count"] = len(self._cache.edges)
        
        # Persist
        self._write_snapshot(self._cache)
        logger.debug(f"Deleted edge from snapshot: {edge_id}")
        return True
    
    def rebuild_indexes(
        self,
        graph_store: "GraphStore",
        vector_store: "VectorStore",
        embedding_service: "EmbeddingService"
    ) -> tuple[int, int]:
        """
        Rebuild Neo4j graph and FAISS index from snapshot.
        
        This is the key recovery mechanism:
        1. Load snapshot (source of truth)
        2. Clear existing Neo4j data
        3. Recreate all nodes in Neo4j
        4. Recreate all edges in Neo4j
        5. Generate embeddings and rebuild FAISS index
        
        Args:
            graph_store: Neo4j graph store instance
            vector_store: FAISS vector store instance
            embedding_service: Embedding service for generating vectors
            
        Returns:
            Tuple of (nodes_rebuilt, edges_rebuilt)
        """
        import time
        logger.info("Starting index rebuild from snapshot...")
        
        # Load snapshot data
        snapshot = self.load_snapshot()
        total_nodes = len(snapshot.nodes)
        total_edges = len(snapshot.edges)
        
        if total_nodes == 0:
            logger.info("Snapshot is empty, nothing to rebuild")
            return 0, 0
        
        logger.info(f"Rebuilding {total_nodes} nodes and {total_edges} edges...")
        
        # Clear existing data in Neo4j
        try:
            graph_store.clear_all()
        except Exception as e:
            logger.warning(f"Failed to clear Neo4j (may not be connected): {e}")
        
        # Clear FAISS index
        vector_store.clear()
        
        nodes_rebuilt = 0
        edges_rebuilt = 0
        start_time = time.time()
        
        # Import here to avoid circular imports
        from app.models.graph import NodeCreate, EdgeCreate
        
        # Rebuild nodes with progress logging
        for i, (node_id, node_data) in enumerate(snapshot.nodes.items()):
            try:
                # Truncate text to max 50000 chars to meet validation
                text = node_data.get("text", "")
                if len(text) > 50000:
                    text = text[:50000]
                
                # Get topic from node_data (top-level) or metadata
                topic = node_data.get("topic", "")
                metadata = node_data.get("metadata", {})
                
                # Ensure topic is in metadata for consistency
                if topic and "topic" not in metadata:
                    metadata["topic"] = topic
                
                # Create node in Neo4j
                node_create = NodeCreate(
                    id=node_id,
                    text=text,
                    metadata=metadata
                )
                graph_store.create_node(node_create)
                
                # Generate embedding and add to FAISS
                embedding = embedding_service.embed(text)
                vector_store.add_embedding(node_id, embedding)
                
                nodes_rebuilt += 1
                
                # Log progress every 100 nodes
                if nodes_rebuilt % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = nodes_rebuilt / elapsed if elapsed > 0 else 0
                    remaining = (total_nodes - nodes_rebuilt) / rate if rate > 0 else 0
                    logger.info(f"  Progress: {nodes_rebuilt}/{total_nodes} nodes ({rate:.1f}/sec, ~{remaining:.0f}s remaining)")
                
            except Exception as e:
                logger.error(f"Failed to rebuild node {node_id}: {e}")
        
        # Rebuild edges
        for edge_data in snapshot.edges:
            try:
                edge_create = EdgeCreate(
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    type=edge_data.get("type", "RELATED_TO"),
                    weight=edge_data.get("weight", 0.5)
                )
                graph_store.create_edge(edge_create)
                edges_rebuilt += 1
                
            except Exception as e:
                logger.error(f"Failed to rebuild edge: {e}")
        
        logger.info(
            f"Index rebuild complete: {nodes_rebuilt} nodes, {edges_rebuilt} edges"
        )
        
        return nodes_rebuilt, edges_rebuilt
    
    def _write_snapshot(self, snapshot: SnapshotData) -> None:
        """
        Write snapshot data to JSON file.
        
        Uses atomic write pattern (write to temp, then rename) for safety.
        
        Args:
            snapshot: The SnapshotData to write
        """
        temp_path = self.snapshot_path.with_suffix('.tmp')
        
        try:
            # Write to temp file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Try atomic rename, fallback to direct write if file is locked
            try:
                temp_path.replace(self.snapshot_path)
            except (OSError, PermissionError) as e:
                logger.warning(f"Atomic rename failed (file may be locked): {e}")
                logger.info("Falling back to direct write...")
                # Fallback: direct write to the target file
                with open(self.snapshot_path, 'w', encoding='utf-8') as f:
                    json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def get_node_count(self) -> int:
        """Get the number of nodes in the snapshot."""
        if self._cache is None:
            self.load_snapshot()
        return len(self._cache.nodes)
    
    def get_edge_count(self) -> int:
        """Get the number of edges in the snapshot."""
        if self._cache is None:
            self.load_snapshot()
        return len(self._cache.edges)
    
    def get_all_nodes(self) -> dict[str, dict[str, Any]]:
        """
        Get all nodes from the snapshot.
        
        Returns:
            Dictionary mapping node_id -> node data
        """
        if self._cache is None:
            self.load_snapshot()
        return self._cache.nodes.copy()
    
    def get_all_edges(self) -> list[dict[str, Any]]:
        """
        Get all edges from the snapshot.
        
        Returns:
            List of edge data dictionaries
        """
        if self._cache is None:
            self.load_snapshot()
        return self._cache.edges.copy()
    
    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """
        Get a single node by ID from the snapshot.
        
        Args:
            node_id: The node ID to look up
            
        Returns:
            Node data dictionary, or None if not found
        """
        if self._cache is None:
            self.load_snapshot()
        return self._cache.nodes.get(node_id)
    
    def get_cached_data(self) -> Optional[SnapshotData]:
        """Get the cached snapshot data (may be None if not loaded)."""
        return self._cache

    def get_aggregated_stats(self) -> dict[str, Any]:
        """
        Get aggregated statistics about the data.
        
        Returns topic distribution, category distribution, edge type distribution,
        and other useful metrics for the Stats page.
        """
        if self._cache is None:
            self.load_snapshot()
        
        # Topic distribution
        topic_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        
        for node_id, node_data in self._cache.nodes.items():
            metadata = node_data.get("metadata", {})
            
            # Count topics - check both top-level and in metadata
            topic = node_data.get("topic") or metadata.get("topic", "unknown")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Count categories
            category = metadata.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count sources
            source = metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Edge type distribution
        edge_type_counts: dict[str, int] = {}
        total_weight = 0.0
        for edge in self._cache.edges:
            edge_type = edge.get("type", "unknown")
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            total_weight += edge.get("weight", 0.0)
        
        # Calculate average edge weight
        avg_edge_weight = total_weight / len(self._cache.edges) if self._cache.edges else 0.0
        
        # Calculate average degree (edges per node)
        avg_degree = (len(self._cache.edges) * 2) / len(self._cache.nodes) if self._cache.nodes else 0.0
        
        return {
            "topic_distribution": [
                {"name": k, "value": v} for k, v in sorted(topic_counts.items(), key=lambda x: -x[1])
            ],
            "category_distribution": [
                {"name": k, "count": v} for k, v in sorted(category_counts.items(), key=lambda x: -x[1])
            ],
            "source_distribution": [
                {"name": k, "count": v} for k, v in sorted(source_counts.items(), key=lambda x: -x[1])
            ],
            "edge_type_distribution": [
                {"name": k, "count": v} for k, v in sorted(edge_type_counts.items(), key=lambda x: -x[1])
            ],
            "avg_edge_weight": round(avg_edge_weight, 4),
            "avg_degree": round(avg_degree, 2),
            "unique_topics": len(topic_counts),
            "unique_categories": len(category_counts),
            "unique_sources": len(source_counts),
        }
