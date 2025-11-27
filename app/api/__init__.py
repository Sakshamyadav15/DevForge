"""
API routes package.

Contains FastAPI routers for:
- /nodes - Node CRUD operations
- /edges - Edge/relationship CRUD operations  
- /search - Vector, graph, and hybrid search endpoints
"""

from app.api.nodes import router as nodes_router
from app.api.edges import router as edges_router
from app.api.search import router as search_router

__all__ = [
    "nodes_router",
    "edges_router",
    "search_router",
]
