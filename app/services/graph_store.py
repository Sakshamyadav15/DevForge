"""
Graph Store Service (Neo4j)

This module provides graph database operations using Neo4j:
- Node CRUD operations
- Edge/relationship CRUD operations
- Graph traversal (multi-hop reasoning)
- Graph statistics for hybrid search scoring

Neo4j Schema:
- Nodes have label :Node with properties {id, text, metadata_json}
- Relationships have type :RELATION with properties {type, weight, id}

Edge Type Weights:
Different relationship types have different importance for ranking.
For example, CITES (direct citation) is stronger than RELATED_TO (general).
These weights are used in graph scoring for hybrid search.
"""

import json
import logging
from typing import Any, Optional
from datetime import datetime
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

from app.config import settings
from app.models.graph import Node, NodeCreate, NodeUpdate, Edge, EdgeCreate, generate_node_id, generate_edge_id

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Neo4j-based graph store for node and relationship management.
    
    This class provides:
    - CRUD operations for nodes and edges
    - Graph traversal for multi-hop reasoning
    - Statistics computation for hybrid search scoring
    
    The store uses Cypher queries for efficient graph operations.
    Nodes are stored with their text and metadata (as JSON string).
    Relationships store type and weight for weighted graph scoring.
    
    Attributes:
        driver: Neo4j driver instance
        database: Name of the Neo4j database to use
        
    Example:
        >>> store = GraphStore()
        >>> node = store.create_node(NodeCreate(text="AI in healthcare", metadata={"topic": "health"}))
        >>> store.create_edge(EdgeCreate(source_id=node.id, target_id="other_node", type="RELATED_TO"))
        >>> neighbors = store.traverse(node.id, depth=2)
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize the graph store with Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (default: from settings)
            user: Neo4j username (default: from settings)
            password: Neo4j password (default: from settings)
            database: Database name (default: from settings)
        """
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self._uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    @contextmanager
    def _session(self):
        """
        Context manager for Neo4j sessions.
        
        Yields:
            Neo4j session for executing queries
        """
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def create_node(self, node_data: NodeCreate) -> Node:
        """
        Create a new node in the graph.
        
        Args:
            node_data: Node creation data (text, metadata, optional id)
            
        Returns:
            The created Node with assigned ID
            
        Raises:
            Neo4jError: If node creation fails
        """
        # Generate ID if not provided
        node_id = node_data.id or generate_node_id()
        now = datetime.utcnow()
        
        # Serialize metadata to JSON string for Neo4j storage
        metadata_json = json.dumps(node_data.metadata)
        
        query = """
        CREATE (n:Node {
            id: $id,
            text: $text,
            metadata_json: $metadata_json,
            created_at: $created_at,
            updated_at: $updated_at
        })
        RETURN n
        """
        
        with self._session() as session:
            result = session.run(
                query,
                id=node_id,
                text=node_data.text,
                metadata_json=metadata_json,
                created_at=now.isoformat(),
                updated_at=now.isoformat()
            )
            record = result.single()
            
            if not record:
                raise Neo4jError("Failed to create node")
            
            logger.debug(f"Created node: {node_id}")
            
            return Node(
                id=node_id,
                text=node_data.text,
                metadata=node_data.metadata,
                created_at=now,
                updated_at=now
            )
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: The unique node identifier
            
        Returns:
            The Node if found, None otherwise
        """
        query = """
        MATCH (n:Node {id: $id})
        RETURN n
        """
        
        with self._session() as session:
            result = session.run(query, id=node_id)
            record = result.single()
            
            if not record:
                return None
            
            return self._record_to_node(record["n"])
    
    def update_node(self, node_id: str, update_data: NodeUpdate) -> Optional[Node]:
        """
        Update an existing node.
        
        Args:
            node_id: The node ID to update
            update_data: Fields to update (text and/or metadata)
            
        Returns:
            The updated Node if found, None otherwise
        """
        # Build dynamic update query based on provided fields
        set_clauses = ["n.updated_at = $updated_at"]
        params: dict[str, Any] = {
            "id": node_id,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if update_data.text is not None:
            set_clauses.append("n.text = $text")
            params["text"] = update_data.text
        
        if update_data.metadata is not None:
            set_clauses.append("n.metadata_json = $metadata_json")
            params["metadata_json"] = json.dumps(update_data.metadata)
        
        query = f"""
        MATCH (n:Node {{id: $id}})
        SET {', '.join(set_clauses)}
        RETURN n
        """
        
        with self._session() as session:
            result = session.run(query, **params)
            record = result.single()
            
            if not record:
                return None
            
            logger.debug(f"Updated node: {node_id}")
            return self._record_to_node(record["n"])
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and all its relationships.
        
        Args:
            node_id: The node ID to delete
            
        Returns:
            True if node was deleted, False if not found
        """
        # DETACH DELETE removes the node and all connected relationships
        query = """
        MATCH (n:Node {id: $id})
        DETACH DELETE n
        RETURN count(n) AS deleted
        """
        
        with self._session() as session:
            result = session.run(query, id=node_id)
            record = result.single()
            deleted = record["deleted"] > 0
            
            if deleted:
                logger.debug(f"Deleted node: {node_id}")
            
            return deleted
    
    def get_all_nodes(self) -> list[Node]:
        """
        Retrieve all nodes from the graph.
        
        Returns:
            List of all Node objects
        """
        query = "MATCH (n:Node) RETURN n"
        
        with self._session() as session:
            result = session.run(query)
            return [self._record_to_node(record["n"]) for record in result]
    
    # =========================================================================
    # Edge Operations
    # =========================================================================
    
    def create_edge(self, edge_data: EdgeCreate) -> Edge:
        """
        Create a relationship between two nodes.
        
        Args:
            edge_data: Edge creation data (source_id, target_id, type, weight)
            
        Returns:
            The created Edge
            
        Raises:
            ValueError: If source or target node doesn't exist
        """
        edge_id = generate_edge_id(edge_data.source_id, edge_data.target_id, edge_data.type)
        now = datetime.utcnow()
        
        # Use MERGE to avoid duplicate edges of same type between same nodes
        query = """
        MATCH (source:Node {id: $source_id})
        MATCH (target:Node {id: $target_id})
        MERGE (source)-[r:RELATION {type: $type}]->(target)
        SET r.id = $id,
            r.weight = $weight,
            r.created_at = $created_at
        RETURN r, source.id AS source_id, target.id AS target_id
        """
        
        with self._session() as session:
            result = session.run(
                query,
                source_id=edge_data.source_id,
                target_id=edge_data.target_id,
                type=edge_data.type,
                weight=edge_data.weight,
                id=edge_id,
                created_at=now.isoformat()
            )
            record = result.single()
            
            if not record:
                raise ValueError(
                    f"Failed to create edge: source ({edge_data.source_id}) "
                    f"or target ({edge_data.target_id}) node not found"
                )
            
            logger.debug(f"Created edge: {edge_id}")
            
            return Edge(
                id=edge_id,
                source_id=edge_data.source_id,
                target_id=edge_data.target_id,
                type=edge_data.type,
                weight=edge_data.weight,
                created_at=now
            )
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """
        Retrieve an edge by ID.
        
        Args:
            edge_id: The unique edge identifier
            
        Returns:
            The Edge if found, None otherwise
        """
        query = """
        MATCH (source:Node)-[r:RELATION {id: $id}]->(target:Node)
        RETURN r, source.id AS source_id, target.id AS target_id
        """
        
        with self._session() as session:
            result = session.run(query, id=edge_id)
            record = result.single()
            
            if not record:
                return None
            
            return self._record_to_edge(record)
    
    def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge by ID.
        
        Args:
            edge_id: The edge ID to delete
            
        Returns:
            True if edge was deleted, False if not found
        """
        query = """
        MATCH ()-[r:RELATION {id: $id}]->()
        DELETE r
        RETURN count(r) AS deleted
        """
        
        with self._session() as session:
            result = session.run(query, id=edge_id)
            record = result.single()
            deleted = record["deleted"] > 0
            
            if deleted:
                logger.debug(f"Deleted edge: {edge_id}")
            
            return deleted
    
    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """
        Get all edges connected to a node (both incoming and outgoing).
        
        Args:
            node_id: The node ID
            
        Returns:
            List of Edge objects
        """
        query = """
        MATCH (n:Node {id: $node_id})-[r:RELATION]-(other:Node)
        WITH r, 
             CASE WHEN startNode(r).id = $node_id THEN startNode(r).id ELSE endNode(r).id END AS source,
             CASE WHEN startNode(r).id = $node_id THEN endNode(r).id ELSE startNode(r).id END AS target
        RETURN r, source AS source_id, target AS target_id
        """
        
        with self._session() as session:
            result = session.run(query, node_id=node_id)
            return [self._record_to_edge(record) for record in result]
    
    def get_all_edges(self) -> list[Edge]:
        """
        Retrieve all edges from the graph.
        
        Returns:
            List of all Edge objects
        """
        query = """
        MATCH (source:Node)-[r:RELATION]->(target:Node)
        RETURN r, source.id AS source_id, target.id AS target_id
        """
        
        with self._session() as session:
            result = session.run(query)
            return [self._record_to_edge(record) for record in result]
    
    # =========================================================================
    # Graph Traversal & Statistics
    # =========================================================================
    
    def get_neighbors_and_stats(self, node_id: str, max_depth: int = 1) -> dict[str, Any]:
        """
        Get graph statistics for a node (used in hybrid search scoring).
        
        This method computes:
        - degree: number of outgoing edges
        - avg_weight: average weight of connected edges
        - type_weighted_sum: sum(edge_weight * EDGE_TYPE_WEIGHTS[type])
        - two_hop_count: number of nodes reachable in 2 hops (if max_depth > 1)
        
        Args:
            node_id: The node to compute stats for
            max_depth: Maximum depth for extended stats (default: 1)
            
        Returns:
            Dictionary with graph statistics
        """
        # Get basic stats (1-hop)
        query = """
        MATCH (n:Node {id: $node_id})
        OPTIONAL MATCH (n)-[r:RELATION]->(neighbor:Node)
        WITH n,
             count(DISTINCT neighbor) AS degree,
             avg(r.weight) AS avg_weight,
             collect({type: r.type, weight: r.weight}) AS edges
        RETURN degree, avg_weight, edges
        """
        
        with self._session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if not record:
                return {
                    "degree": 0,
                    "avg_weight": 0.0,
                    "type_weighted_sum": 0.0,
                    "two_hop_count": 0
                }
            
            degree = record["degree"] or 0
            avg_weight = record["avg_weight"] or 0.0
            edges = record["edges"] or []
            
            # Compute type-weighted sum using edge type weights
            type_weighted_sum = 0.0
            edge_type_weights = settings.edge_type_weights
            
            for edge in edges:
                if edge and edge.get("type") and edge.get("weight"):
                    edge_type = edge["type"]
                    edge_weight = edge["weight"]
                    type_weight = edge_type_weights.get(edge_type, 0.5)
                    type_weighted_sum += edge_weight * type_weight
            
            stats = {
                "degree": degree,
                "avg_weight": avg_weight,
                "type_weighted_sum": type_weighted_sum,
                "two_hop_count": 0
            }
            
            # Get 2-hop count if requested
            if max_depth > 1:
                two_hop_query = """
                MATCH (n:Node {id: $node_id})-[:RELATION*2]->(hop2:Node)
                WHERE hop2.id <> $node_id
                RETURN count(DISTINCT hop2) AS two_hop_count
                """
                two_hop_result = session.run(two_hop_query, node_id=node_id)
                two_hop_record = two_hop_result.single()
                if two_hop_record:
                    stats["two_hop_count"] = two_hop_record["two_hop_count"] or 0
            
            return stats
    
    def traverse(self, start_id: str, depth: int = 1) -> list[Node]:
        """
        Perform multi-hop traversal from a starting node.
        
        This is the core multi-hop reasoning capability, allowing discovery
        of indirectly connected nodes.
        
        Args:
            start_id: ID of the starting node
            depth: Maximum number of hops (1-5)
            
        Returns:
            List of nodes reachable within the specified depth
        """
        # Clamp depth to reasonable range
        depth = max(1, min(5, depth))
        
        # Variable-length path pattern: *1..depth
        query = """
        MATCH (start:Node {id: $start_id})
        OPTIONAL MATCH (start)-[:RELATION*1..""" + str(depth) + """]->(reachable:Node)
        WHERE reachable.id <> $start_id
        RETURN DISTINCT reachable
        """
        
        with self._session() as session:
            result = session.run(query, start_id=start_id)
            nodes = []
            
            for record in result:
                if record["reachable"]:
                    nodes.append(self._record_to_node(record["reachable"]))
            
            logger.debug(f"Traversal from {start_id} (depth={depth}): found {len(nodes)} nodes")
            return nodes
    
    def traverse_with_distances(
        self,
        start_id: str,
        depth: int = 1
    ) -> list[tuple[Node, int, float]]:
        """
        Traverse with hop distance and path weight information.
        
        Args:
            start_id: ID of the starting node
            depth: Maximum number of hops
            
        Returns:
            List of (node, hop_distance, path_weight) tuples
        """
        depth = max(1, min(5, depth))
        
        query = """
        MATCH (start:Node {id: $start_id})
        OPTIONAL MATCH path = (start)-[:RELATION*1..""" + str(depth) + """]->(reachable:Node)
        WHERE reachable.id <> $start_id
        WITH reachable, 
             min(length(path)) AS hop_distance,
             [r IN relationships(path) | r.weight] AS weights
        RETURN DISTINCT reachable, hop_distance, 
               reduce(acc = 0.0, w IN weights | acc + COALESCE(w, 0)) AS path_weight
        """
        
        with self._session() as session:
            result = session.run(query, start_id=start_id)
            nodes = []
            
            for record in result:
                if record["reachable"]:
                    node = self._record_to_node(record["reachable"])
                    hop_distance = record["hop_distance"] or 0
                    path_weight = record["path_weight"] or 0.0
                    nodes.append((node, hop_distance, path_weight))
            
            return nodes
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear_all(self) -> None:
        """
        Delete all nodes and relationships from the graph.
        
        Use with caution - this removes ALL data.
        """
        query = "MATCH (n) DETACH DELETE n"
        
        with self._session() as session:
            session.run(query)
            logger.warning("Cleared all nodes and relationships from graph")
    
    def _record_to_node(self, node_data: Any) -> Node:
        """
        Convert a Neo4j node record to a Node model.
        
        Args:
            node_data: Neo4j node object
            
        Returns:
            Node model instance
        """
        # Parse metadata from JSON string
        metadata_json = node_data.get("metadata_json", "{}")
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}
        
        # Parse timestamps
        created_at = None
        updated_at = None
        
        if node_data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(node_data["created_at"])
            except (ValueError, TypeError):
                pass
        
        if node_data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(node_data["updated_at"])
            except (ValueError, TypeError):
                pass
        
        return Node(
            id=node_data["id"],
            text=node_data["text"],
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at
        )
    
    def _record_to_edge(self, record: Any) -> Edge:
        """
        Convert a Neo4j relationship record to an Edge model.
        
        Args:
            record: Neo4j record with r, source_id, target_id
            
        Returns:
            Edge model instance
        """
        rel = record["r"]
        
        created_at = None
        if rel.get("created_at"):
            try:
                created_at = datetime.fromisoformat(rel["created_at"])
            except (ValueError, TypeError):
                pass
        
        return Edge(
            id=rel.get("id", ""),
            source_id=record["source_id"],
            target_id=record["target_id"],
            type=rel.get("type", "RELATED_TO"),
            weight=rel.get("weight", 0.5),
            created_at=created_at
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
