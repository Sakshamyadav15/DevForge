"""
Seed Data Script

This script populates the knowledge graph with sample data for demonstration.
It creates nodes representing research topics and connects them with relationships.

Dataset Theme: AI and Healthcare Research
- Nodes represent research topics, papers, and concepts
- Edges represent relationships like RELATED_TO, CITES, SUPPORTS

Usage:
    python -m scripts.seed_data
    
Or via API (if server is running):
    curl -X POST http://localhost:8000/seed

This demonstrates:
1. How nodes and edges are created
2. The types of relationships in the graph
3. Data suitable for demonstrating hybrid search
"""

import requests
import time
from typing import Optional

# Base URL of the running API
BASE_URL = "http://localhost:8000"


# =============================================================================
# Sample Dataset
# =============================================================================

SAMPLE_NODES = [
    {
        "id": "ai_healthcare_overview",
        "text": "Artificial Intelligence in Healthcare: A comprehensive overview of how AI technologies are transforming medical diagnosis, treatment planning, and patient care. Machine learning models can analyze medical images, predict disease progression, and assist in drug discovery.",
        "metadata": {"source": "review_paper", "topic": "healthcare", "year": 2024}
    },
    {
        "id": "medical_imaging_ai",
        "text": "Deep Learning for Medical Imaging: Convolutional neural networks have achieved remarkable success in analyzing X-rays, CT scans, and MRI images. These models can detect tumors, fractures, and other abnormalities with accuracy comparable to radiologists.",
        "metadata": {"source": "research_paper", "topic": "medical_imaging", "year": 2023}
    },
    {
        "id": "nlp_clinical_notes",
        "text": "Natural Language Processing for Clinical Notes: NLP techniques enable extraction of structured information from unstructured clinical notes. Named entity recognition and relation extraction help identify diagnoses, medications, and symptoms from doctor's notes.",
        "metadata": {"source": "research_paper", "topic": "nlp", "year": 2023}
    },
    {
        "id": "drug_discovery_ml",
        "text": "Machine Learning in Drug Discovery: AI accelerates drug discovery by predicting molecular properties, identifying drug targets, and optimizing lead compounds. Graph neural networks model molecular structures for property prediction.",
        "metadata": {"source": "industry_report", "topic": "drug_discovery", "year": 2024}
    },
    {
        "id": "predictive_analytics",
        "text": "Predictive Analytics for Patient Outcomes: Statistical and machine learning models predict patient readmission risk, disease progression, and treatment response. Early warning systems help clinicians intervene before adverse events.",
        "metadata": {"source": "clinical_study", "topic": "predictive_analytics", "year": 2023}
    },
    {
        "id": "cnn_fundamentals",
        "text": "Convolutional Neural Networks Fundamentals: CNNs are specialized neural networks for processing grid-like data such as images. They use convolutional layers to extract hierarchical features, pooling for spatial invariance, and fully connected layers for classification.",
        "metadata": {"source": "textbook", "topic": "deep_learning", "year": 2022}
    },
    {
        "id": "transformer_architecture",
        "text": "Transformer Architecture and Attention Mechanisms: Transformers revolutionized NLP with self-attention mechanisms that capture long-range dependencies. They form the basis of models like BERT, GPT, and have been adapted for computer vision.",
        "metadata": {"source": "seminal_paper", "topic": "deep_learning", "year": 2017}
    },
    {
        "id": "federated_learning_healthcare",
        "text": "Federated Learning for Healthcare: Privacy-preserving machine learning that enables training models across hospitals without sharing patient data. Each institution trains locally and only shares model updates, protecting sensitive health information.",
        "metadata": {"source": "research_paper", "topic": "privacy", "year": 2024}
    },
    {
        "id": "explainable_ai_medical",
        "text": "Explainable AI in Medical Decision Making: XAI techniques provide interpretable explanations for AI predictions in healthcare. Methods like SHAP values, attention visualization, and concept-based explanations help clinicians trust and validate AI recommendations.",
        "metadata": {"source": "review_paper", "topic": "explainability", "year": 2023}
    },
    {
        "id": "radiology_automation",
        "text": "Automated Radiology Reporting: AI systems that automatically generate preliminary radiology reports from medical images. These systems combine image analysis with natural language generation to produce structured findings.",
        "metadata": {"source": "clinical_study", "topic": "medical_imaging", "year": 2024}
    },
    {
        "id": "genomics_ml",
        "text": "Machine Learning for Genomics: Deep learning models analyze genomic sequences for variant calling, gene expression prediction, and disease risk assessment. Attention mechanisms help identify important genetic markers.",
        "metadata": {"source": "research_paper", "topic": "genomics", "year": 2023}
    },
    {
        "id": "wearable_health_ai",
        "text": "AI for Wearable Health Devices: Machine learning algorithms process data from smartwatches and fitness trackers to detect arrhythmias, predict falls, and monitor chronic conditions in real-time.",
        "metadata": {"source": "industry_report", "topic": "wearables", "year": 2024}
    },
    {
        "id": "clinical_trial_optimization",
        "text": "AI-Powered Clinical Trial Design: Machine learning optimizes patient recruitment, trial site selection, and protocol design. Predictive models improve trial success rates by identifying suitable patient populations.",
        "metadata": {"source": "industry_report", "topic": "clinical_trials", "year": 2023}
    },
    {
        "id": "mental_health_nlp",
        "text": "NLP for Mental Health Assessment: Natural language processing analyzes speech patterns and written text to detect depression, anxiety, and other mental health conditions. Sentiment analysis and linguistic features provide objective markers.",
        "metadata": {"source": "research_paper", "topic": "mental_health", "year": 2024}
    },
    {
        "id": "surgical_robotics_ai",
        "text": "AI in Robotic Surgery: Machine learning enhances surgical robots with better motion planning, tremor compensation, and autonomous suturing. Computer vision guides instruments with sub-millimeter precision.",
        "metadata": {"source": "clinical_study", "topic": "surgery", "year": 2023}
    }
]

SAMPLE_EDGES = [
    # AI Healthcare overview is a central hub
    {"source_id": "ai_healthcare_overview", "target_id": "medical_imaging_ai", "type": "RELATED_TO", "weight": 0.9},
    {"source_id": "ai_healthcare_overview", "target_id": "nlp_clinical_notes", "type": "RELATED_TO", "weight": 0.8},
    {"source_id": "ai_healthcare_overview", "target_id": "drug_discovery_ml", "type": "RELATED_TO", "weight": 0.85},
    {"source_id": "ai_healthcare_overview", "target_id": "predictive_analytics", "type": "RELATED_TO", "weight": 0.9},
    
    # Medical imaging connections
    {"source_id": "medical_imaging_ai", "target_id": "cnn_fundamentals", "type": "CITES", "weight": 1.0},
    {"source_id": "medical_imaging_ai", "target_id": "radiology_automation", "type": "SUPPORTS", "weight": 0.9},
    {"source_id": "medical_imaging_ai", "target_id": "explainable_ai_medical", "type": "RELATED_TO", "weight": 0.7},
    
    # NLP connections
    {"source_id": "nlp_clinical_notes", "target_id": "transformer_architecture", "type": "CITES", "weight": 1.0},
    {"source_id": "nlp_clinical_notes", "target_id": "radiology_automation", "type": "SUPPORTS", "weight": 0.8},
    {"source_id": "nlp_clinical_notes", "target_id": "mental_health_nlp", "type": "RELATED_TO", "weight": 0.75},
    
    # Drug discovery connections
    {"source_id": "drug_discovery_ml", "target_id": "genomics_ml", "type": "RELATED_TO", "weight": 0.85},
    {"source_id": "drug_discovery_ml", "target_id": "clinical_trial_optimization", "type": "SUPPORTS", "weight": 0.8},
    
    # Deep learning foundation connections
    {"source_id": "cnn_fundamentals", "target_id": "transformer_architecture", "type": "RELATED_TO", "weight": 0.6},
    
    # Privacy and explainability
    {"source_id": "federated_learning_healthcare", "target_id": "ai_healthcare_overview", "type": "SUPPORTS", "weight": 0.85},
    {"source_id": "explainable_ai_medical", "target_id": "ai_healthcare_overview", "type": "SUPPORTS", "weight": 0.9},
    
    # Predictive analytics connections
    {"source_id": "predictive_analytics", "target_id": "wearable_health_ai", "type": "RELATED_TO", "weight": 0.8},
    {"source_id": "predictive_analytics", "target_id": "clinical_trial_optimization", "type": "SUPPORTS", "weight": 0.7},
    
    # Genomics connections
    {"source_id": "genomics_ml", "target_id": "transformer_architecture", "type": "CITES", "weight": 0.9},
    
    # Surgery and robotics
    {"source_id": "surgical_robotics_ai", "target_id": "medical_imaging_ai", "type": "RELATED_TO", "weight": 0.7},
    {"source_id": "surgical_robotics_ai", "target_id": "cnn_fundamentals", "type": "CITES", "weight": 0.8},
    
    # Wearables connections
    {"source_id": "wearable_health_ai", "target_id": "federated_learning_healthcare", "type": "RELATED_TO", "weight": 0.75},
]


# =============================================================================
# Seeding Functions
# =============================================================================

def create_node(node_data: dict) -> Optional[dict]:
    """Create a single node via API."""
    try:
        response = requests.post(f"{BASE_URL}/nodes", json=node_data)
        if response.status_code == 201:
            return response.json()
        else:
            print(f"Failed to create node {node_data.get('id')}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error - is the server running at {BASE_URL}?")
        return None


def create_edge(edge_data: dict) -> Optional[dict]:
    """Create a single edge via API."""
    try:
        response = requests.post(f"{BASE_URL}/edges", json=edge_data)
        if response.status_code == 201:
            return response.json()
        else:
            print(f"Failed to create edge: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        return None


def seed_database():
    """Seed the database with sample data."""
    print("=" * 60)
    print("DevForge - Seeding Database with Sample Data")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"Server health check failed: {response.text}")
            return
        print("Server is healthy, starting seed...")
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to server at {BASE_URL}")
        print("Please start the server first: uvicorn app.main:app --reload")
        return
    
    # Create nodes
    print("\nCreating nodes...")
    nodes_created = 0
    for node_data in SAMPLE_NODES:
        result = create_node(node_data)
        if result:
            nodes_created += 1
            print(f"  ✓ Created: {result['id']}")
        time.sleep(0.1)  # Small delay to avoid overwhelming the server
    
    print(f"\nNodes created: {nodes_created}/{len(SAMPLE_NODES)}")
    
    # Create edges
    print("\nCreating edges...")
    edges_created = 0
    for edge_data in SAMPLE_EDGES:
        result = create_edge(edge_data)
        if result:
            edges_created += 1
            print(f"  ✓ Created: {edge_data['source_id']} --[{edge_data['type']}]--> {edge_data['target_id']}")
        time.sleep(0.1)
    
    print(f"\nEdges created: {edges_created}/{len(SAMPLE_EDGES)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Seeding Complete!")
    print("=" * 60)
    print(f"\nTotal nodes: {nodes_created}")
    print(f"Total edges: {edges_created}")
    print("\nYou can now test the search endpoints:")
    print("  - POST /search/vector - Vector-only search")
    print("  - POST /search/hybrid - Hybrid search")
    print("  - GET /search/graph?start_id=ai_healthcare_overview&depth=2 - Graph traversal")
    print("  - GET /search/compare?query_text=medical%20imaging - Compare search methods")


def test_searches():
    """Run some example searches to demonstrate the system."""
    print("\n" + "=" * 60)
    print("Running Example Searches")
    print("=" * 60)
    
    # Test 1: Vector search
    print("\n1. Vector Search: 'deep learning for medical diagnosis'")
    response = requests.post(f"{BASE_URL}/search/vector", json={
        "query_text": "deep learning for medical diagnosis",
        "top_k": 3
    })
    if response.status_code == 200:
        results = response.json()
        for r in results["results"]:
            print(f"   [{r['rank']}] {r['node']['id']} (similarity: {r['cosine_similarity']:.3f})")
    
    # Test 2: Hybrid search
    print("\n2. Hybrid Search: 'connection between AI and drug discovery'")
    response = requests.post(f"{BASE_URL}/search/hybrid", json={
        "query_text": "connection between AI and drug discovery",
        "top_k": 3
    })
    if response.status_code == 200:
        results = response.json()
        print(f"   Weights used: vector={results['vector_weight_used']:.2f}, graph={results['graph_weight_used']:.2f}")
        print(f"   Ranking changed from vector-only: {results['ranking_changed']}")
        for r in results["results"]:
            print(f"   [{r['rank']}] {r['node']['id']} (final: {r['final_score']:.3f}, vec: {r['cosine_similarity']:.3f}, graph: {r['graph_score']:.3f})")
    
    # Test 3: Graph traversal
    print("\n3. Graph Traversal from 'ai_healthcare_overview' (depth=2)")
    response = requests.get(f"{BASE_URL}/search/graph?start_id=ai_healthcare_overview&depth=2")
    if response.status_code == 200:
        results = response.json()
        print(f"   Found {results['total_nodes']} connected nodes")
        for node in results["traversed_nodes"][:5]:
            print(f"   - {node['node']['id']} (hop: {node['hop_distance']})")
    
    # Test 4: Compare search
    print("\n4. Compare Vector vs Hybrid: 'transformer models in healthcare'")
    response = requests.get(f"{BASE_URL}/search/compare?query_text=transformer%20models%20in%20healthcare&top_k=3")
    if response.status_code == 200:
        results = response.json()
        print(f"   Ranking changes: {results['statistics']['ranking_changes']}")
        print("   Vector-only ranking:")
        for r in results["vector_only_results"]:
            print(f"      - {r['node_id']} ({r['cosine_similarity']:.3f})")
        print("   Hybrid ranking:")
        for r in results["hybrid_results"]:
            print(f"      - {r['node_id']} (vec_rank: {r['vector_only_rank']} → hybrid_rank: {r['hybrid_rank']})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_searches()
    else:
        seed_database()
        
        # Ask if user wants to run test searches
        print("\nWould you like to run test searches? (y/n): ", end="")
        try:
            answer = input().strip().lower()
            if answer == 'y':
                test_searches()
        except EOFError:
            pass
