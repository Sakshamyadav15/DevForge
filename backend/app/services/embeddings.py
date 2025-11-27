"""
Embedding Service

This module provides text-to-vector embedding functionality using:
1. SentenceTransformer (all-MiniLM-L6-v2) as the primary model
2. A deterministic mock fallback for testing/offline environments

Why all-MiniLM-L6-v2?
- Excellent speed/accuracy tradeoff - produces 384-dimensional embeddings
- CPU-friendly - no GPU required, suitable for hackathons and local dev
- Well-suited for semantic similarity tasks
- Produces normalized vectors (important for cosine similarity)
- Small model size (~80MB) - fast to download and load

Why have a mock fallback?
- Robustness: allows the system to run without the model
- Testing: deterministic embeddings make unit tests reproducible
- Offline development: work without downloading the model
- CI/CD: run tests in environments without model access
"""

import logging
import hashlib
import numpy as np
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def mock_embed(text: str, dim: int = 384) -> np.ndarray:
    """
    Generate a deterministic mock embedding for testing purposes.
    
    Uses the hash of the text as a seed for reproducible random vectors.
    This ensures the same text always produces the same embedding,
    which is essential for testing and debugging.
    
    Args:
        text: The input text to "embed"
        dim: The embedding dimension (default: 384 to match all-MiniLM-L6-v2)
        
    Returns:
        A normalized numpy array of shape (dim,)
    """
    # Use MD5 hash of text as seed for reproducibility
    # MD5 is fine here - we just need determinism, not security
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    seed = int(text_hash[:8], 16)  # Use first 8 hex chars as seed
    
    # Create random generator with seed
    rng = np.random.default_rng(seed)
    
    # Generate random vector
    embedding = rng.random(dim).astype(np.float32)
    
    # Normalize to unit length (important for cosine similarity)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    This service wraps the embedding model and provides a simple interface
    for generating embeddings from text. It handles model loading and
    provides a mock fallback when the model cannot be loaded.
    
    Attributes:
        model: The SentenceTransformer model instance (or None if using mock)
        use_mock: Whether mock embeddings are being used
        _dimension: The embedding dimension
        
    Example:
        >>> service = EmbeddingService()
        >>> embedding = service.embed("AI is transforming healthcare")
        >>> print(embedding.shape)  # (384,)
        >>> print(np.linalg.norm(embedding))  # ~1.0 (normalized)
    """
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize the embedding service.
        
        Args:
            use_mock: If True, always use mock embeddings. If False,
                     attempt to load the real model and fall back to
                     mock if loading fails.
        """
        self.model = None
        self.use_mock = use_mock or settings.use_mock_embeddings
        self._dimension = settings.embedding_dimension
        
        if not self.use_mock:
            self._load_model()
        else:
            logger.info("Using mock embedding service (configured)")
    
    def _load_model(self) -> None:
        """
        Attempt to load the SentenceTransformer model.
        
        Falls back to mock embeddings if loading fails for any reason
        (model not found, insufficient memory, etc.)
        """
        try:
            # Import here to avoid loading the library if not needed
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {settings.embedding_model_name}")
            self.model = SentenceTransformer(settings.embedding_model_name)
            
            # Verify the model dimension matches our config
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            actual_dim = test_embedding.shape[0]
            
            if actual_dim != self._dimension:
                logger.warning(
                    f"Model dimension ({actual_dim}) differs from config ({self._dimension}). "
                    f"Using model dimension."
                )
                self._dimension = actual_dim
            
            logger.info(f"Embedding model loaded successfully (dim={self._dimension})")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to mock embeddings. "
                "Install with: pip install sentence-transformers"
            )
            self.use_mock = True
            
        except Exception as e:
            logger.warning(
                f"Failed to load embedding model: {e}. "
                "Falling back to mock embeddings."
            )
            self.use_mock = True
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the given text.
        
        The returned vector is normalized to unit length, suitable for
        cosine similarity computation using inner product.
        
        Args:
            text: The input text to embed
            
        Returns:
            A numpy array of shape (dimension,) with dtype float32
            
        Example:
            >>> embedding = service.embed("Hello world")
            >>> embedding.shape
            (384,)
        """
        if self.use_mock:
            return mock_embed(text, self._dimension)
        
        # Use SentenceTransformer to generate embedding
        # normalize_embeddings=True ensures unit vectors
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Batch processing is more efficient than calling embed() repeatedly
        because the model can process multiple inputs in parallel.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        
        if self.use_mock:
            # Generate mock embeddings for each text
            return np.array([mock_embed(t, self._dimension) for t in texts])
        
        # Use batch encoding for efficiency
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100  # Show progress for large batches
        )
        
        return embeddings.astype(np.float32)
    
    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            The number of dimensions in each embedding vector
        """
        return self._dimension
    
    @property
    def is_mock(self) -> bool:
        """
        Check if the service is using mock embeddings.
        
        Returns:
            True if using mock embeddings, False if using the real model
        """
        return self.use_mock
    
    def __repr__(self) -> str:
        """String representation of the service."""
        mode = "mock" if self.use_mock else settings.embedding_model_name
        return f"EmbeddingService(mode={mode}, dim={self._dimension})"
