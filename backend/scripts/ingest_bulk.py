#!/usr/bin/env python3
"""
DevForge Bulk Data Ingestion Script

Ingests large datasets into DevForge via direct snapshot writing.
Supports: HuggingFace datasets (arxiv, pubmed), arXiv Kaggle, custom JSON.

Usage:
    # AI/ML papers (25K)
    python -m scripts.ingest_bulk -l 25000 -t arxiv
    
    # Medical papers (25K)  
    python -m scripts.ingest_bulk -l 25000 -t pubmed
    
    # From local arXiv JSON
    python -m scripts.ingest_bulk --source arxiv --file path/to/arxiv.json -l 25000
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
from dataclasses import dataclass

# Paths - Use absolute path relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "snapshot.json"
DEVFORGE_API_URL = "http://localhost:8000"

# AI/ML categories for filtering
ARXIV_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "cs.RO",
    "stat.ML", "cs.IR", "cs.MA", "cs.HC"
]

# Edge types
EDGE_TYPES = ["CITES", "RELATED_TO", "SIMILAR_TO", "REFERENCES", "BUILDS_ON"]


@dataclass
class Document:
    """Represents a document to ingest."""
    id: str
    text: str
    metadata: dict


# =============================================================================
# Data Source: HuggingFace
# =============================================================================

def load_from_huggingface(limit: int = 25000, topic: str = "arxiv") -> Generator[Document, None, None]:
    """
    Load documents from HuggingFace datasets.
    
    Uses modern Parquet-based datasets (no custom scripts required).
    
    Args:
        limit: Maximum number of documents
        topic: "arxiv" for AI/ML papers, "pubmed" for medical papers
        
    Yields:
        Document objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return
    
    # Map topics to datasets
    if topic == "arxiv":
        dataset_name = "ccdv/arxiv-summarization"
        text_field = "article"
        abstract_field = "abstract"
    elif topic == "pubmed":
        dataset_name = "ccdv/pubmed-summarization"
        text_field = "article"
        abstract_field = "abstract"
    else:
        print(f"Unknown topic: {topic}. Use 'arxiv' or 'pubmed'.")
        return
    
    print(f"Loading {limit} documents from HuggingFace {dataset_name}...")
    print("(This may take a few minutes on first run - downloading dataset)")
    
    try:
        # Use streaming to avoid memory issues
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    count = 0
    for i, item in enumerate(dataset):
        if count >= limit:
            break
        
        doc_id = f"hf_{topic}_{i:06d}"
        
        # Get text and abstract
        text = str(item.get(text_field, ""))
        abstract = str(item.get(abstract_field, ""))
        
        if abstract:
            text = f"{abstract}\n\n{text}"
        
        # Truncate to max 45000 chars (leaving room for safety margin)
        text = text[:45000]
        
        if len(text) < 100:
            continue
        
        yield Document(
            id=doc_id,
            text=text,
            metadata={
                "source": "huggingface",
                "dataset": dataset_name,
                "topic": topic,
                "index": i
            }
        )
        
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count}/{limit} documents...")


# =============================================================================
# Data Source: arXiv Kaggle Dataset
# =============================================================================

def load_from_arxiv_kaggle(
    file_path: str,
    limit: int = 25000,
    categories: Optional[list] = None
) -> Generator[Document, None, None]:
    """
    Load documents from arXiv Kaggle dataset (JSON lines format).
    
    Download from: https://www.kaggle.com/datasets/Cornell-University/arxiv
    """
    categories = categories or ARXIV_CATEGORIES
    categories_set = set(categories)
    
    print(f"Loading from arXiv dataset: {file_path}")
    print(f"Filtering categories: {categories}")
    
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if count >= limit:
                break
            
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            paper_categories = paper.get("categories", "").split()
            if not any(cat in categories_set for cat in paper_categories):
                continue
            
            arxiv_id = paper.get("id", f"arxiv_{line_num}")
            title = paper.get("title", "").replace("\n", " ")
            abstract = paper.get("abstract", "").replace("\n", " ")
            
            if len(abstract) < 50:
                continue
            
            text = f"{title}\n\n{abstract}"
            
            yield Document(
                id=f"arxiv_{arxiv_id}",
                text=text,
                metadata={
                    "source": "arxiv_kaggle",
                    "arxiv_id": arxiv_id,
                    "categories": paper_categories,
                    "authors": paper.get("authors", ""),
                    "year": paper.get("update_date", "")[:4] if paper.get("update_date") else None,
                    "topic": "arxiv"
                }
            )
            
            count += 1
            if count % 5000 == 0:
                print(f"  Processed {count}/{limit} documents...")


# =============================================================================
# Data Source: Custom JSON
# =============================================================================

def load_from_json(file_path: str, limit: int = 25000) -> Generator[Document, None, None]:
    """Load from custom JSON file with documents array."""
    print(f"Loading from JSON: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = data if isinstance(data, list) else data.get("documents", [])
    
    for i, doc in enumerate(documents[:limit]):
        doc_id = doc.get("id", f"json_{i:06d}")
        text = doc.get("text", doc.get("content", doc.get("body", "")))
        
        if len(text) < 50:
            continue
        
        yield Document(
            id=doc_id,
            text=text[:5000],
            metadata=doc.get("metadata", {"source": "json", "index": i})
        )
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1} documents...")


# =============================================================================
# Snapshot Writer
# =============================================================================

def write_to_snapshot(
    documents: Generator[Document, None, None],
    limit: int,
    create_edges: bool = True
) -> tuple[int, int]:
    """
    Write documents directly to snapshot file.
    UPDATES existing data - does not overwrite.
    """
    print("\n" + "=" * 60)
    print("Writing to Snapshot (Direct Mode)")
    print("=" * 60)
    
    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    
    # Load existing snapshot or create new
    if SNAPSHOT_PATH.exists():
        print(f"Loading existing snapshot from {SNAPSHOT_PATH}...")
        with open(SNAPSHOT_PATH, 'r') as f:
            snapshot = json.load(f)
        print(f"  Existing nodes: {len(snapshot.get('nodes', {})):,}")
        print(f"  Existing edges: {len(snapshot.get('edges', [])):,}")
    else:
        snapshot = {"nodes": {}, "edges": [], "metadata": {}}
    
    nodes = snapshot.get("nodes", {})
    edges = snapshot.get("edges", [])
    
    initial_node_count = len(nodes)
    initial_edge_count = len(edges)
    
    new_node_ids = []
    
    print("\nProcessing documents...")
    start_time = time.time()
    
    for i, doc in enumerate(documents):
        if i >= limit:
            break
        
        # Only add if not already exists
        if doc.id not in nodes:
            nodes[doc.id] = {
                "text": doc.text,
                "metadata": doc.metadata,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            new_node_ids.append(doc.id)
        
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1:,} documents ({rate:.1f} docs/sec)")
    
    nodes_created = len(new_node_ids)
    print(f"\nNew nodes added: {nodes_created:,}")
    print(f"Total nodes now: {len(nodes):,}")
    
    # Create edges between NEW documents only
    edges_created = 0
    if create_edges and len(new_node_ids) > 1:
        print("\nCreating edges for new documents...")
        
        # Group by topic
        topic_groups = {}
        for node_id in new_node_ids:
            topic = nodes[node_id]["metadata"].get("topic", "general")
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(node_id)
        
        # Create edges within topic groups
        for topic, group_ids in topic_groups.items():
            if len(group_ids) < 2:
                continue
            
            # Sequential connections
            for j in range(min(len(group_ids) - 1, 5000)):
                edges.append({
                    "source_id": group_ids[j],
                    "target_id": group_ids[j + 1],
                    "type": "RELATED_TO",
                    "weight": round(random.uniform(0.6, 0.9), 3),
                    "metadata": {"auto_generated": True, "topic": topic}
                })
                edges_created += 1
            
            # Random connections within topic
            sample_size = min(len(group_ids), 2000)
            sampled = random.sample(group_ids, sample_size)
            for j in range(0, len(sampled) - 1, 2):
                edge_type = random.choice(EDGE_TYPES)
                edges.append({
                    "source_id": sampled[j],
                    "target_id": sampled[j + 1],
                    "type": edge_type,
                    "weight": round(random.uniform(0.5, 1.0), 3),
                    "metadata": {"auto_generated": True, "topic": topic}
                })
                edges_created += 1
        
        print(f"Edges created: {edges_created:,}")
    
    # Save snapshot
    print("\nSaving snapshot...")
    snapshot["nodes"] = nodes
    snapshot["edges"] = edges
    snapshot["metadata"] = {
        "last_updated": datetime.utcnow().isoformat(),
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }
    
    with open(SNAPSHOT_PATH, 'w') as f:
        json.dump(snapshot, f)
    
    size_mb = SNAPSHOT_PATH.stat().st_size / (1024 * 1024)
    print(f"Snapshot saved: {SNAPSHOT_PATH} ({size_mb:.1f} MB)")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f} seconds")
    
    return nodes_created, edges_created


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DevForge Bulk Ingestion")
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=25000,
        help="Maximum documents to ingest (default: 25000)"
    )
    parser.add_argument(
        "-s", "--source",
        choices=["huggingface", "arxiv", "json"],
        default="huggingface",
        help="Data source (default: huggingface)"
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to data file (required for arxiv/json sources)"
    )
    parser.add_argument(
        "-t", "--topic",
        default="arxiv",
        help="HuggingFace topic: 'arxiv' (AI/ML) or 'pubmed' (medical)"
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Skip creating edges"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"DevForge Bulk Ingestion - {args.limit:,} Documents")
    print(f"Topic: {args.topic}")
    print("=" * 60)
    
    # Select data source
    if args.source == "huggingface":
        documents = load_from_huggingface(args.limit, args.topic)
    elif args.source == "arxiv":
        if not args.file:
            print("ERROR: --file required for arxiv source")
            return
        documents = load_from_arxiv_kaggle(args.file, args.limit)
    elif args.source == "json":
        if not args.file:
            print("ERROR: --file required for json source")
            return
        documents = load_from_json(args.file, args.limit)
    else:
        print(f"Unknown source: {args.source}")
        return
    
    # Ingest
    nodes, edges = write_to_snapshot(
        documents,
        limit=args.limit,
        create_edges=not args.no_edges
    )
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"New nodes added: {nodes:,}")
    print(f"New edges added: {edges:,}")
    print(f"\nSnapshot: {SNAPSHOT_PATH}")
    print("\nNext: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
