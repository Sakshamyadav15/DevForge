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


class DocumentIngestRequest(BaseModel):
    """Request model for ingesting unstructured documents that get chunked into entities."""
    content: str = Field(..., description="The full document/text content to be processed", min_length=10)
    title: Optional[str] = Field(default=None, description="Document title")
    source: Optional[str] = Field(default=None, description="Source of the document")
    topic: Optional[str] = Field(default=None, description="Topic/category of the document")


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: str
    title: str
    chunks_created: int
    nodes: List[Dict[str, Any]]
    edges_created: int
    message: str


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
    import app.main as main_module
    
    embedding_service = getattr(main_module, 'embedding_service', None)
    vector_store = getattr(main_module, 'vector_store', None)
    graph_store = getattr(main_module, 'graph_store', None)
    snapshot_manager = getattr(main_module, 'snapshot_manager', None)
    
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


# =============================================================================
# Document Chunking Utilities
# =============================================================================

def chunk_document(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split a document into overlapping chunks.
    
    Uses sentence boundaries when possible for cleaner chunks.
    """
    import re
    
    # Clean the text
    text = text.strip()
    if not text:
        return []
    
    # Try to split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current and start new
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from the end of current chunk
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:] + " " + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If we only got one chunk that's still too big, split by sentences
    if len(chunks) == 1 and len(chunks[0]) > chunk_size * 1.5:
        text = chunks[0]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def extract_entities_and_topics(text: str) -> Dict[str, Any]:
    """
    Extract key entities and infer topic from text using simple NLP.
    """
    import re
    
    # Common topic keywords
    topic_keywords = {
        "technology": ["software", "programming", "code", "algorithm", "computer", "data", "api", "system", "database", "server", "cloud", "ai", "machine learning", "neural", "network"],
        "science": ["research", "study", "experiment", "theory", "hypothesis", "scientific", "laboratory", "discovery", "physics", "chemistry", "biology"],
        "business": ["company", "market", "revenue", "profit", "customer", "product", "service", "management", "strategy", "investment", "finance"],
        "healthcare": ["medical", "health", "patient", "doctor", "hospital", "treatment", "disease", "medicine", "clinical", "diagnosis"],
        "education": ["learning", "student", "teacher", "school", "university", "course", "training", "knowledge", "education", "study"],
    }
    
    text_lower = text.lower()
    
    # Detect topic
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            topic_scores[topic] = score
    
    detected_topic = max(topic_scores, key=topic_scores.get) if topic_scores else "general"
    
    # Extract potential entities (capitalized words that aren't at sentence start)
    words = text.split()
    entities = []
    for i, word in enumerate(words):
        # Skip first word of sentences and common words
        if i > 0 and word[0].isupper() and len(word) > 2:
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word.lower() not in ["the", "this", "that", "these", "those", "and", "but", "for"]:
                entities.append(clean_word)
    
    # Get unique entities
    entities = list(set(entities))[:10]  # Limit to 10 entities
    
    return {
        "topic": detected_topic,
        "entities": entities,
        "word_count": len(words)
    }


@router.post("/document", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    """
    Ingest an unstructured document by:
    1. Automatically chunking the document into optimal entities/nodes
    2. Generating embeddings for each chunk
    3. Creating nodes in Neo4j for each chunk
    4. Adding embeddings to FAISS
    5. Creating SEQUENTIAL relationships between chunks (NEXT_CHUNK)
    6. Creating SEMANTIC relationships based on similarity
    7. Persisting to snapshot.json
    
    The system automatically determines optimal chunk size based on document length.
    """
    embedding_service, vector_store, graph_store, snapshot_manager = get_services()
    
    import uuid
    
    # Generate document ID
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    doc_title = request.title or request.content[:50].strip() + "..."
    topic = request.topic
    source = request.source or "document_upload"
    
    # Extract topic if not provided
    if not topic:
        extracted = extract_entities_and_topics(request.content)
        topic = extracted["topic"]
    
    # SMART CHUNKING: Automatically determine optimal chunk size based on document length
    doc_length = len(request.content)
    if doc_length < 500:
        # Very short document - treat as single node
        chunk_size = doc_length + 1
        chunk_overlap = 0
    elif doc_length < 2000:
        # Short document - larger chunks, minimal overlap
        chunk_size = 500
        chunk_overlap = 50
    elif doc_length < 10000:
        # Medium document - balanced chunks
        chunk_size = 400
        chunk_overlap = 80
    else:
        # Large document - smaller chunks for better granularity
        chunk_size = 350
        chunk_overlap = 100
    
    try:
        # 1. Chunk the document with smart settings
        logger.info(f"Chunking document '{doc_title}' (length={doc_length}) with auto chunk_size={chunk_size}, overlap={chunk_overlap}")
        chunks = chunk_document(request.content, chunk_size, chunk_overlap)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document could not be chunked - too short or empty")
        
        logger.info(f"Created {len(chunks)} chunks from document")
        
        created_nodes = []
        node_embeddings = []
        total_edges_created = 0
        
        # 2. Process each chunk
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i:03d}"
            chunk_title = f"{doc_title} [Part {i+1}/{len(chunks)}]"
            
            # Extract entities from this chunk
            chunk_info = extract_entities_and_topics(chunk_text)
            
            # Build metadata
            chunk_metadata = {
                "title": chunk_title,
                "document_id": doc_id,
                "document_title": doc_title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "topic": topic,
                "entities": chunk_info["entities"],
                "word_count": chunk_info["word_count"],
            }
            
            # Generate embedding
            logger.info(f"Generating embedding for chunk {i+1}/{len(chunks)}")
            embedding = embedding_service.embed(chunk_text)
            
            # Create node in Neo4j
            node = graph_store.create_node(NodeCreate(
                text=chunk_text,
                metadata={
                    "title": chunk_title,
                    "topic": topic,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "category": "document_chunk",
                    "source": source,
                }
            ))
            node_id = node.id
            logger.info(f"Created node {node_id} in Neo4j")
            
            # Add to FAISS
            vector_store.add_embedding(node_id, embedding)
            
            # Save to snapshot
            snapshot_node_data = {
                "text": chunk_text,
                "topic": topic,
                "metadata": chunk_metadata
            }
            update_snapshot_with_node(node_id, snapshot_node_data)
            
            # Track for relationship creation
            created_nodes.append({
                "id": node_id,
                "chunk_index": i,
                "text": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                "topic": topic,
                "metadata": chunk_metadata
            })
            node_embeddings.append((node_id, embedding, topic))
        
        # 3. Create SEQUENTIAL relationships (NEXT_CHUNK) between consecutive chunks
        logger.info("Creating sequential relationships between chunks")
        for i in range(len(created_nodes) - 1):
            try:
                from app.models.graph import EdgeCreate
                edge = graph_store.create_edge(EdgeCreate(
                    source_id=created_nodes[i]["id"],
                    target_id=created_nodes[i+1]["id"],
                    type="NEXT_CHUNK",
                    weight=1.0  # Strong sequential connection
                ))
                update_snapshot_with_edge({
                    "id": edge.id,
                    "source_id": created_nodes[i]["id"],
                    "target_id": created_nodes[i+1]["id"],
                    "type": "NEXT_CHUNK",
                    "weight": 1.0
                })
                total_edges_created += 1
            except Exception as e:
                logger.warning(f"Failed to create sequential edge: {e}")
        
        # 4. Create SEMANTIC relationships based on similarity
        logger.info("Creating semantic relationships between chunks")
        for node_id, embedding, node_topic in node_embeddings:
            try:
                edges = find_and_create_relationships(
                    node_id=node_id,
                    node_text="",
                    node_topic=node_topic,
                    embedding=embedding,
                    vector_store=vector_store,
                    graph_store=graph_store,
                    similarity_threshold=0.5,  # Higher threshold for same-document chunks
                    max_edges=2
                )
                total_edges_created += edges
            except Exception as e:
                logger.warning(f"Failed to create semantic relationships for {node_id}: {e}")
        
        # 5. Save FAISS index
        save_faiss_index(vector_store)
        
        logger.info(f"Successfully ingested document '{doc_title}' with {len(created_nodes)} chunks and {total_edges_created} edges")
        
        return DocumentIngestResponse(
            document_id=doc_id,
            title=doc_title,
            chunks_created=len(created_nodes),
            nodes=created_nodes,
            edges_created=total_edges_created,
            message=f"Document '{doc_title}' successfully ingested. Created {len(created_nodes)} chunks with {total_edges_created} relationships."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )
