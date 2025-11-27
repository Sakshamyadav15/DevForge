#!/usr/bin/env python3
"""
DevForge Sample Data Generator

Generates synthetic but realistic knowledge graph data for testing.
Creates concise nodes with short, meaningful text (2-5 sentences).

Usage:
    python -m scripts.generate_data -l 5000
    python -m scripts.generate_data -l 1000 --topic tech
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "snapshot.json"

# Edge types
EDGE_TYPES = ["RELATED_TO", "CITES", "EXTENDS", "SUPPORTS", "CONTRADICTS", "SIMILAR_TO"]

# =============================================================================
# Knowledge Base Templates
# =============================================================================

TECH_TOPICS = {
    "machine_learning": [
        "Machine learning enables computers to learn patterns from data without explicit programming.",
        "Supervised learning uses labeled data to train models for prediction tasks.",
        "Unsupervised learning discovers hidden patterns in unlabeled data.",
        "Deep learning uses neural networks with multiple layers to learn complex representations.",
        "Reinforcement learning trains agents through reward-based feedback.",
        "Transfer learning applies knowledge from one domain to solve problems in another.",
        "Feature engineering transforms raw data into meaningful inputs for ML models.",
        "Cross-validation helps evaluate model performance and prevent overfitting.",
        "Gradient descent optimizes model parameters by minimizing loss functions.",
        "Ensemble methods combine multiple models to improve prediction accuracy.",
    ],
    "neural_networks": [
        "Neural networks are computational models inspired by biological neurons.",
        "Convolutional neural networks excel at image recognition tasks.",
        "Recurrent neural networks process sequential data like text and time series.",
        "Transformers use attention mechanisms to capture long-range dependencies.",
        "Backpropagation computes gradients for training neural networks.",
        "Activation functions introduce non-linearity into neural networks.",
        "Dropout regularization prevents overfitting by randomly disabling neurons.",
        "Batch normalization stabilizes training by normalizing layer inputs.",
        "Skip connections in ResNets enable training of very deep networks.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
    ],
    "nlp": [
        "Natural language processing enables computers to understand human language.",
        "Word embeddings represent words as dense vectors capturing semantic meaning.",
        "Named entity recognition identifies people, places, and organizations in text.",
        "Sentiment analysis determines the emotional tone of text content.",
        "Machine translation converts text between different languages.",
        "Text summarization condenses long documents into shorter versions.",
        "Question answering systems extract answers from text passages.",
        "Language models predict the next word in a sequence.",
        "Tokenization splits text into words or subword units for processing.",
        "Part-of-speech tagging labels words with their grammatical categories.",
    ],
    "computer_vision": [
        "Computer vision enables machines to interpret visual information.",
        "Image classification assigns category labels to entire images.",
        "Object detection locates and identifies multiple objects in images.",
        "Semantic segmentation labels each pixel with its object class.",
        "Image generation creates new images using generative models.",
        "Face recognition identifies individuals from facial features.",
        "Optical character recognition converts images of text to digital text.",
        "Pose estimation determines the position of body joints in images.",
        "Depth estimation infers 3D distance information from 2D images.",
        "Image super-resolution enhances image quality and resolution.",
    ],
    "databases": [
        "Relational databases store data in structured tables with defined schemas.",
        "NoSQL databases provide flexible schemas for unstructured data.",
        "Graph databases model relationships between entities as edges and nodes.",
        "Vector databases enable similarity search using embedding representations.",
        "Database indexing speeds up query performance significantly.",
        "ACID properties ensure reliable database transactions.",
        "Database sharding distributes data across multiple servers.",
        "Query optimization improves database performance through efficient execution plans.",
        "Caching reduces database load by storing frequently accessed data in memory.",
        "Database replication provides redundancy and improves read performance.",
    ],
    "cloud_computing": [
        "Cloud computing delivers computing resources over the internet on demand.",
        "Infrastructure as a Service provides virtualized computing resources.",
        "Platform as a Service offers development and deployment environments.",
        "Serverless computing executes code without managing servers.",
        "Container orchestration manages deployment of containerized applications.",
        "Auto-scaling adjusts resources based on demand automatically.",
        "Cloud storage provides scalable and durable data storage.",
        "Content delivery networks distribute content globally for faster access.",
        "Load balancing distributes traffic across multiple servers.",
        "Cloud security protects data and applications in cloud environments.",
    ],
    "web_development": [
        "Frontend development creates user interfaces for web applications.",
        "Backend development handles server-side logic and data processing.",
        "RESTful APIs enable communication between web services.",
        "GraphQL provides flexible queries for client-server communication.",
        "Single-page applications update content without full page reloads.",
        "Progressive web apps combine web and mobile app features.",
        "Web frameworks accelerate development with pre-built components.",
        "Responsive design adapts layouts to different screen sizes.",
        "Web accessibility ensures applications are usable by everyone.",
        "Performance optimization improves web application speed and efficiency.",
    ],
    "cybersecurity": [
        "Encryption protects data by converting it to unreadable format.",
        "Authentication verifies the identity of users and systems.",
        "Authorization controls access to resources based on permissions.",
        "Firewalls filter network traffic to prevent unauthorized access.",
        "Intrusion detection systems monitor networks for suspicious activity.",
        "Vulnerability scanning identifies security weaknesses in systems.",
        "Penetration testing simulates attacks to find security flaws.",
        "Security audits evaluate compliance with security policies.",
        "Incident response handles security breaches and threats.",
        "Zero trust security verifies every access request regardless of source.",
    ],
    "data_engineering": [
        "ETL processes extract, transform, and load data between systems.",
        "Data pipelines automate the flow of data through processing stages.",
        "Data warehouses store structured data for analytical queries.",
        "Data lakes store raw data in various formats at scale.",
        "Stream processing handles real-time data flows continuously.",
        "Batch processing handles large volumes of data at scheduled intervals.",
        "Data quality ensures accuracy and consistency of data.",
        "Data governance manages data assets and policies.",
        "Schema design structures data for efficient storage and retrieval.",
        "Data versioning tracks changes to datasets over time.",
    ],
    "devops": [
        "Continuous integration automatically builds and tests code changes.",
        "Continuous deployment automatically releases validated changes to production.",
        "Infrastructure as code manages infrastructure through configuration files.",
        "Monitoring tracks application performance and health metrics.",
        "Logging captures events and errors for debugging and analysis.",
        "Configuration management maintains consistent system configurations.",
        "Version control tracks changes to code and enables collaboration.",
        "Automated testing validates code quality through test suites.",
        "Microservices architecture divides applications into small independent services.",
        "Service mesh manages communication between microservices.",
    ],
}

SCIENCE_TOPICS = {
    "physics": [
        "Quantum mechanics describes behavior of matter at atomic scales.",
        "Relativity explains gravity as curvature of spacetime.",
        "Thermodynamics studies heat, energy, and their transformations.",
        "Electromagnetism unifies electricity and magnetism.",
        "Classical mechanics describes motion of macroscopic objects.",
        "Particle physics studies fundamental particles and forces.",
        "Condensed matter physics explores properties of solid and liquid matter.",
        "Astrophysics applies physics to understand celestial objects.",
        "Nuclear physics studies atomic nuclei and their interactions.",
        "Optics investigates the behavior and properties of light.",
    ],
    "biology": [
        "Genetics studies heredity and variation in living organisms.",
        "Evolution explains how species change over time through natural selection.",
        "Cell biology examines the structure and function of cells.",
        "Molecular biology studies biological processes at the molecular level.",
        "Ecology investigates interactions between organisms and their environment.",
        "Neuroscience explores the structure and function of the nervous system.",
        "Immunology studies the immune system and its responses.",
        "Microbiology examines microscopic organisms and their effects.",
        "Biochemistry studies chemical processes within living organisms.",
        "Developmental biology explores how organisms grow and develop.",
    ],
    "chemistry": [
        "Organic chemistry studies carbon-containing compounds.",
        "Inorganic chemistry examines non-carbon compounds and metals.",
        "Physical chemistry applies physics to understand chemical systems.",
        "Analytical chemistry develops methods to determine composition.",
        "Biochemistry bridges chemistry and biology in living systems.",
        "Polymer chemistry studies large molecules made of repeating units.",
        "Environmental chemistry examines chemical processes in the environment.",
        "Medicinal chemistry designs and develops pharmaceutical compounds.",
        "Electrochemistry studies chemical reactions involving electricity.",
        "Catalysis accelerates chemical reactions using catalysts.",
    ],
    "mathematics": [
        "Calculus studies continuous change through derivatives and integrals.",
        "Linear algebra examines vector spaces and linear transformations.",
        "Probability theory quantifies uncertainty and random events.",
        "Statistics analyzes and interprets numerical data.",
        "Number theory studies properties of integers and prime numbers.",
        "Topology examines properties preserved under continuous deformations.",
        "Graph theory studies networks of nodes and edges.",
        "Optimization finds the best solution from available alternatives.",
        "Differential equations model systems that change over time.",
        "Abstract algebra studies algebraic structures like groups and rings.",
    ],
}

BUSINESS_TOPICS = {
    "management": [
        "Strategic planning defines long-term organizational goals and approaches.",
        "Project management coordinates resources to achieve specific objectives.",
        "Change management guides organizations through transitions.",
        "Risk management identifies and mitigates potential problems.",
        "Quality management ensures products meet defined standards.",
        "Operations management oversees production and service delivery.",
        "Human resource management handles employee recruitment and development.",
        "Financial management plans and controls organizational finances.",
        "Supply chain management coordinates the flow of goods and services.",
        "Performance management evaluates and improves employee effectiveness.",
    ],
    "marketing": [
        "Market research gathers information about customer needs and preferences.",
        "Brand management builds and maintains brand identity and value.",
        "Digital marketing promotes products through online channels.",
        "Content marketing attracts customers through valuable content.",
        "Social media marketing engages audiences on social platforms.",
        "Search engine optimization improves website visibility in search results.",
        "Email marketing communicates with customers through targeted messages.",
        "Customer segmentation divides markets into distinct groups.",
        "Marketing analytics measures and optimizes campaign performance.",
        "Product positioning differentiates offerings in the marketplace.",
    ],
    "finance": [
        "Investment analysis evaluates opportunities for financial returns.",
        "Portfolio management balances risk and return across assets.",
        "Financial modeling projects future financial performance.",
        "Valuation determines the economic worth of assets or companies.",
        "Capital budgeting evaluates long-term investment decisions.",
        "Corporate finance manages company funding and financial structure.",
        "Risk assessment quantifies potential financial losses.",
        "Financial reporting communicates financial status to stakeholders.",
        "Budgeting allocates resources to achieve organizational goals.",
        "Cash flow management ensures adequate liquidity for operations.",
    ],
}

def generate_nodes(count: int, topic_filter: str = None) -> list[dict]:
    """Generate synthetic nodes with short, meaningful text."""
    
    # Combine topic categories
    all_topics = {}
    if topic_filter == "tech" or topic_filter is None:
        all_topics.update(TECH_TOPICS)
    if topic_filter == "science" or topic_filter is None:
        all_topics.update(SCIENCE_TOPICS)
    if topic_filter == "business" or topic_filter is None:
        all_topics.update(BUSINESS_TOPICS)
    
    nodes = []
    topic_names = list(all_topics.keys())
    
    for i in range(count):
        # Select random topic
        topic = random.choice(topic_names)
        statements = all_topics[topic]
        
        # Create node with 2-4 related sentences
        num_sentences = random.randint(2, 4)
        selected = random.sample(statements, min(num_sentences, len(statements)))
        text = " ".join(selected)
        
        # Add some variation
        prefixes = [
            "", "", "",  # Most nodes have no prefix
            "Research shows that ",
            "Studies indicate that ",
            "It is well established that ",
            "Recent findings suggest that ",
            "Experts agree that ",
        ]
        prefix = random.choice(prefixes)
        if prefix:
            text = prefix.lower() + text[0].lower() + text[1:]
        
        node_id = f"node_{i:05d}"
        nodes.append({
            "id": node_id,
            "text": text,
            "metadata": {
                "topic": topic,
                "category": get_category(topic),
                "generated": True,
                "index": i
            }
        })
    
    return nodes

def get_category(topic: str) -> str:
    """Get the category for a topic."""
    if topic in TECH_TOPICS:
        return "technology"
    elif topic in SCIENCE_TOPICS:
        return "science"
    elif topic in BUSINESS_TOPICS:
        return "business"
    return "general"

def generate_edges(nodes: list[dict], edge_ratio: float = 0.5) -> list[dict]:
    """Generate edges between related nodes."""
    
    edges = []
    node_ids = [n["id"] for n in nodes]
    
    # Group nodes by topic for more meaningful connections
    topic_groups = {}
    for node in nodes:
        topic = node["metadata"]["topic"]
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(node["id"])
    
    num_edges = int(len(nodes) * edge_ratio)
    
    for i in range(num_edges):
        # 70% within same topic, 30% across topics
        if random.random() < 0.7:
            # Same topic connection
            topic = random.choice(list(topic_groups.keys()))
            group = topic_groups[topic]
            if len(group) >= 2:
                source, target = random.sample(group, 2)
            else:
                source, target = random.sample(node_ids, 2)
        else:
            # Cross-topic connection
            source, target = random.sample(node_ids, 2)
        
        edge_type = random.choice(EDGE_TYPES)
        weight = round(random.uniform(0.5, 1.0), 3)
        
        edges.append({
            "source_id": source,
            "target_id": target,
            "type": edge_type,
            "weight": weight,
            "metadata": {"auto_generated": True}
        })
    
    return edges

def save_to_snapshot(nodes: list[dict], edges: list[dict]) -> None:
    """Save generated data to snapshot file."""
    
    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    
    # Convert nodes list to dict
    nodes_dict = {}
    for node in nodes:
        node_id = node["id"]
        nodes_dict[node_id] = {
            "text": node["text"],
            "metadata": node["metadata"],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    snapshot = {
        "nodes": nodes_dict,
        "edges": edges,
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }
    }
    
    with open(SNAPSHOT_PATH, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    size_kb = SNAPSHOT_PATH.stat().st_size / 1024
    print(f"\nSnapshot saved: {SNAPSHOT_PATH}")
    print(f"Size: {size_kb:.1f} KB")

def main():
    parser = argparse.ArgumentParser(description="Generate sample data for DevForge")
    parser.add_argument("-l", "--limit", type=int, default=5000, help="Number of nodes to generate")
    parser.add_argument("--topic", choices=["tech", "science", "business"], help="Filter by topic category")
    parser.add_argument("--edge-ratio", type=float, default=0.5, help="Edges per node ratio")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"DevForge Data Generator")
    print("=" * 60)
    print(f"Generating {args.limit} nodes...")
    
    nodes = generate_nodes(args.limit, args.topic)
    print(f"Generated {len(nodes)} nodes")
    
    print(f"Generating edges (ratio: {args.edge_ratio})...")
    edges = generate_edges(nodes, args.edge_ratio)
    print(f"Generated {len(edges)} edges")
    
    # Show sample
    print("\n--- Sample Node ---")
    sample = random.choice(nodes)
    print(f"ID: {sample['id']}")
    print(f"Topic: {sample['metadata']['topic']}")
    print(f"Text: {sample['text'][:200]}...")
    
    save_to_snapshot(nodes, edges)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"\nNext: uvicorn app.main:app --reload --port 8000")

if __name__ == "__main__":
    main()
