"""
Vector Store Service (FAISS)

This module provides vector storage and similarity search using FAISS.
It implements:
- In-memory vector index with ID mapping
- Cosine similarity search (via inner product on normalized vectors)
- Add/delete operations for dynamic updates
- Persistence to disk (save/load)

FAISS Configuration:
- Uses IndexFlatIP (Inner Product) for exact search
- Wrapped in IndexIDMap to map string node IDs to numeric FAISS IDs
- Vectors must be normalized for cosine similarity via inner product

Why FAISS?
- Extremely fast similarity search (millions of vectors in milliseconds)
- Supports both exact and approximate search
- CPU-optimized (no GPU required)
- Simple Python API

Note: For larger datasets (>1M vectors), consider using IVF indices for
approximate search with better memory/speed tradeoffs.
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for embedding storage and similarity search.
    
    This class manages:
    - A FAISS index for fast similarity search
    - Mapping between string node IDs and numeric FAISS IDs
    - Add/delete/search operations
    
    The store uses inner product similarity, which equals cosine similarity
    when vectors are normalized to unit length.
    
    Attributes:
        dimension: The embedding dimension
        index: The FAISS index (IndexIDMap wrapping IndexFlatIP)
        id_to_faiss: Mapping from string node_id to numeric FAISS ID
        faiss_to_id: Reverse mapping from FAISS ID to node_id
        next_faiss_id: Counter for generating unique FAISS IDs
        
    Example:
        >>> store = VectorStore(dim=384)
        >>> store.add_embedding("node_1", embedding_vector)
        >>> results = store.search(query_vector, top_k=5)
        >>> print(results)  # [("node_1", 0.95), ("node_2", 0.87), ...]
    """
    
    def __init__(self, dim: int):
        """
        Initialize the vector store with the given dimension.
        
        Creates a FAISS IndexFlatIP (Inner Product) index wrapped in
        IndexIDMap for ID-based access.
        
        Args:
            dim: The embedding dimension (must match embeddings being added)
        """
        self.dimension = dim
        
        # Create base index using Inner Product (equals cosine for normalized vectors)
        # IndexFlatIP performs exact search - suitable for < 100k vectors
        base_index = faiss.IndexFlatIP(dim)
        
        # Wrap in IndexIDMap to support custom numeric IDs
        # This lets us delete specific vectors and map to string IDs
        self.index = faiss.IndexIDMap(base_index)
        
        # Mappings between string node IDs and numeric FAISS IDs
        # FAISS requires int64 IDs, so we maintain a bidirectional mapping
        self.id_to_faiss: dict[str, int] = {}
        self.faiss_to_id: dict[int, str] = {}
        
        # Counter for generating unique numeric IDs
        self.next_faiss_id: int = 0
        
        logger.info(f"VectorStore initialized with dimension {dim}")
    
    def add_embedding(self, node_id: str, embedding: np.ndarray) -> None:
        """
        Add an embedding vector for a node.
        
        If the node already exists, its embedding is updated (removed and re-added).
        The embedding is normalized before storage to ensure valid cosine similarity.
        
        Args:
            node_id: Unique string identifier for the node
            embedding: Numpy array of shape (dimension,)
            
        Raises:
            ValueError: If embedding dimension doesn't match store dimension
        """
        # Validate embedding dimension
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} doesn't match "
                f"store dimension {self.dimension}"
            )
        
        # If node already exists, remove old embedding first
        if node_id in self.id_to_faiss:
            self.delete_embedding(node_id)
        
        # Normalize embedding to unit length for cosine similarity
        # Inner product of unit vectors = cosine similarity
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Reshape to (1, dim) for FAISS
        embedding = embedding.reshape(1, -1)
        
        # Assign numeric ID and add to index
        faiss_id = self.next_faiss_id
        self.next_faiss_id += 1
        
        # Add to FAISS with the numeric ID
        self.index.add_with_ids(embedding, np.array([faiss_id], dtype=np.int64))
        
        # Update mappings
        self.id_to_faiss[node_id] = faiss_id
        self.faiss_to_id[faiss_id] = node_id
        
        logger.debug(f"Added embedding for node '{node_id}' (FAISS ID: {faiss_id})")
    
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Retrieve the embedding for a node.
        
        Args:
            node_id: The string ID of the node
            
        Returns:
            The embedding vector as numpy array, or None if not found
        """
        if node_id not in self.id_to_faiss:
            return None
        
        faiss_id = self.id_to_faiss[node_id]
        
        # Reconstruct the embedding from FAISS
        try:
            embedding = self.index.reconstruct(faiss_id)
            return embedding
        except Exception as e:
            logger.warning(f"Could not reconstruct embedding for {node_id}: {e}")
            return None
    
    def delete_embedding(self, node_id: str) -> bool:
        """
        Remove an embedding from the store.
        
        Args:
            node_id: The string ID of the node to remove
            
        Returns:
            True if the embedding was removed, False if it didn't exist
        """
        if node_id not in self.id_to_faiss:
            logger.debug(f"Node '{node_id}' not found in vector store")
            return False
        
        faiss_id = self.id_to_faiss[node_id]
        
        # Remove from FAISS index
        # Note: IndexIDMap supports remove_ids
        self.index.remove_ids(np.array([faiss_id], dtype=np.int64))
        
        # Clean up mappings
        del self.id_to_faiss[node_id]
        del self.faiss_to_id[faiss_id]
        
        logger.debug(f"Removed embedding for node '{node_id}'")
        return True
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Search for the most similar vectors to the query.
        
        Uses inner product similarity, which equals cosine similarity
        for normalized vectors. Results are sorted by score descending.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            top_k: Maximum number of results to return
            
        Returns:
            List of (node_id, cosine_similarity) tuples sorted by score desc.
            Similarity scores are in range [0, 1] for normalized vectors.
            
        Example:
            >>> results = store.search(query_vec, top_k=3)
            >>> for node_id, score in results:
            ...     print(f"{node_id}: {score:.3f}")
        """
        # Handle empty index
        if self.index.ntotal == 0:
            logger.debug("Search on empty index, returning empty results")
            return []
        
        # Clamp top_k to index size
        top_k = min(top_k, self.index.ntotal)
        
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Reshape for FAISS: (1, dim)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Perform search
        # scores: inner product scores (= cosine similarity for unit vectors)
        # ids: FAISS numeric IDs
        scores, ids = self.index.search(query_embedding, top_k)
        
        # Convert results to (node_id, score) tuples
        results = []
        for i in range(len(ids[0])):
            faiss_id = int(ids[0][i])
            score = float(scores[0][i])
            
            # FAISS returns -1 for missing results (when k > ntotal)
            if faiss_id == -1:
                continue
            
            # Look up string node ID
            node_id = self.faiss_to_id.get(faiss_id)
            if node_id:
                # Clamp score to [0, 1] range (can be slightly > 1 due to float precision)
                score = max(0.0, min(1.0, score))
                results.append((node_id, score))
        
        return results
    
    def get_all_node_ids(self) -> list[str]:
        """
        Get all node IDs currently in the store.
        
        Returns:
            List of all string node IDs
        """
        return list(self.id_to_faiss.keys())
    
    def contains(self, node_id: str) -> bool:
        """
        Check if a node exists in the store.
        
        Args:
            node_id: The node ID to check
            
        Returns:
            True if the node exists, False otherwise
        """
        return node_id in self.id_to_faiss
    
    def clear(self) -> None:
        """
        Remove all embeddings from the store.
        
        This resets the index to an empty state.
        """
        # Recreate the index
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        
        # Clear mappings
        self.id_to_faiss.clear()
        self.faiss_to_id.clear()
        
        # Don't reset next_faiss_id to avoid ID reuse issues
        
        logger.info("VectorStore cleared")
    
    @property
    def size(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            Number of embeddings stored
        """
        return self.index.ntotal
    
    def save(self, path: str) -> None:
        """
        Save the FAISS index and ID mappings to disk.
        
        Args:
            path: Base path for saving (will create .faiss and .json files)
        """
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = str(base_path) + ".faiss"
        faiss.write_index(self.index, faiss_path)
        
        # Save ID mappings
        mappings_path = str(base_path) + ".mappings.json"
        mappings = {
            "id_to_faiss": self.id_to_faiss,
            "faiss_to_id": {str(k): v for k, v in self.faiss_to_id.items()},
            "next_faiss_id": self.next_faiss_id,
            "dimension": self.dimension
        }
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f)
        
        logger.info(f"VectorStore saved to {base_path} ({self.size} vectors)")
    
    def load(self, path: str) -> bool:
        """
        Load the FAISS index and ID mappings from disk.
        
        Args:
            path: Base path for loading (expects .faiss and .json files)
            
        Returns:
            True if loaded successfully, False if files don't exist
        """
        faiss_path = str(path) + ".faiss"
        mappings_path = str(path) + ".mappings.json"
        
        if not Path(faiss_path).exists() or not Path(mappings_path).exists():
            logger.debug(f"No saved index found at {path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            
            # Load ID mappings
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
            
            self.id_to_faiss = mappings["id_to_faiss"]
            self.faiss_to_id = {int(k): v for k, v in mappings["faiss_to_id"].items()}
            self.next_faiss_id = mappings["next_faiss_id"]
            
            logger.info(f"VectorStore loaded from {path} ({self.size} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load VectorStore: {e}")
            return False
    
    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self.size
    
    def __bool__(self) -> bool:
        """Return True if the store is initialized (even if empty)."""
        return self.index is not None
    
    def __repr__(self) -> str:
        """String representation of the store."""
        return f"VectorStore(dim={self.dimension}, size={self.size})"
