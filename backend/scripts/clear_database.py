"""
Clear Database Script

This script deletes ALL nodes and relationships from the Neo4j database.
Use with caution - this action is irreversible!

Usage:
    python -m scripts.clear_database
    python -m scripts.clear_database --confirm  # Skip confirmation prompt
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from app.config import settings


def get_node_count(driver, database: str) -> int:
    """Get total number of nodes in the database."""
    with driver.session(database=database) as session:
        result = session.run("MATCH (n) RETURN count(n) AS count")
        record = result.single()
        return record["count"] if record else 0


def get_edge_count(driver, database: str) -> int:
    """Get total number of relationships in the database."""
    with driver.session(database=database) as session:
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
        record = result.single()
        return record["count"] if record else 0


def clear_database(driver, database: str, batch_size: int = 10000) -> tuple[int, int]:
    """
    Delete all nodes and relationships from the database.
    
    Uses batched deletion to avoid memory issues with large datasets.
    
    Args:
        driver: Neo4j driver instance
        database: Database name
        batch_size: Number of nodes to delete per batch
        
    Returns:
        Tuple of (nodes_deleted, relationships_deleted)
    """
    # Get initial counts
    initial_nodes = get_node_count(driver, database)
    initial_edges = get_edge_count(driver, database)
    
    print(f"\n📊 Current database stats:")
    print(f"   Nodes: {initial_nodes:,}")
    print(f"   Relationships: {initial_edges:,}")
    
    if initial_nodes == 0:
        print("\n✅ Database is already empty!")
        return 0, 0
    
    print(f"\n🗑️  Deleting all data in batches of {batch_size:,}...")
    
    total_deleted = 0
    batch_num = 0
    
    with driver.session(database=database) as session:
        while True:
            batch_num += 1
            
            # Delete nodes in batches using DETACH DELETE (removes relationships too)
            result = session.run(f"""
                MATCH (n)
                WITH n LIMIT {batch_size}
                DETACH DELETE n
                RETURN count(n) AS deleted
            """)
            
            record = result.single()
            deleted = record["deleted"] if record else 0
            
            if deleted == 0:
                break
                
            total_deleted += deleted
            print(f"   Batch {batch_num}: Deleted {deleted:,} nodes (Total: {total_deleted:,})")
    
    # Verify deletion
    remaining_nodes = get_node_count(driver, database)
    remaining_edges = get_edge_count(driver, database)
    
    print(f"\n✅ Deletion complete!")
    print(f"   Nodes deleted: {initial_nodes:,}")
    print(f"   Relationships deleted: {initial_edges:,}")
    print(f"   Remaining nodes: {remaining_nodes:,}")
    print(f"   Remaining relationships: {remaining_edges:,}")
    
    return initial_nodes, initial_edges


def main():
    parser = argparse.ArgumentParser(description="Clear all data from Neo4j database")
    parser.add_argument(
        "--confirm", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10000,
        help="Number of nodes to delete per batch (default: 10000)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🗄️  Neo4j Database Cleaner")
    print("=" * 60)
    print(f"\nConnecting to: {settings.neo4j_uri}")
    print(f"Database: {settings.neo4j_database}")
    
    # Connect to Neo4j
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        driver.verify_connectivity()
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)
    
    try:
        # Get current counts
        node_count = get_node_count(driver, settings.neo4j_database)
        edge_count = get_edge_count(driver, settings.neo4j_database)
        
        if node_count == 0 and edge_count == 0:
            print("\n✅ Database is already empty!")
            return
        
        # Confirm deletion
        if not args.confirm:
            print(f"\n⚠️  WARNING: This will delete:")
            print(f"   • {node_count:,} nodes")
            print(f"   • {edge_count:,} relationships")
            print(f"\n   This action is IRREVERSIBLE!")
            
            response = input("\n   Type 'DELETE' to confirm: ").strip()
            
            if response != "DELETE":
                print("\n❌ Aborted. No data was deleted.")
                sys.exit(0)
        
        # Clear the database
        clear_database(driver, settings.neo4j_database, args.batch_size)
        
    finally:
        driver.close()
        print("\n🔌 Connection closed")


if __name__ == "__main__":
    main()
