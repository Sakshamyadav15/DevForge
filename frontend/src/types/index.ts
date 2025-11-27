export interface Node {
  id: string;
  title?: string;
  text?: string;
  topic?: string;
  category?: string;
  created_at: string;
  metadata?: Record<string, any>;
}

export interface NodeCreate {
  text: string;
  metadata?: {
    title?: string;
    topic?: string;
    category?: string;
    source?: string;
    [key: string]: any;
  };
  id?: string;
}

export interface NodeUpdate {
  text?: string;
  metadata?: Record<string, any>;
  regenerate_embedding?: boolean;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight?: number;
  topic?: string;
  created_at: string;
}

export interface EdgeCreate {
  source_id: string;
  target_id: string;
  type: string;
  weight?: number;
}

export interface SearchResult {
  // Format from POST /search/hybrid with adaptive weights
  id: string;
  text: string;
  topic?: string;
  metadata?: Record<string, any>;
  cosine_sim: number;      // Raw cosine similarity
  graph_score: number;     // Normalized graph score
  final_score: number;     // Combined score
  rank: number;
  vector_only_rank?: number;
  degree?: number;
  // Legacy compatibility
  title?: string;
  text_snippet?: string;
  score?: number;          // Alias for final_score
  vector_score?: number;   // Alias for cosine_sim
  neighbors?: number;      // Alias for degree
}

// Legacy format from /hybrid endpoint (if needed)
export interface DetailedSearchResult {
  node: Node;
  cosine_similarity: number;
  cosine_normalized: number;
  graph_score: number;
  graph_normalized: number;
  final_score: number;
  rank: number;
  vector_only_rank: number;
  degree: number;
  avg_edge_weight: number;
}

export interface ChartDataItem {
  name: string;
  value?: number;
  count?: number;
}

export interface Stats {
  nodes: number;
  edges: number;
  snapshot_last_updated: string;
  vector_index_size?: number;
  // Aggregated distributions
  topic_distribution: ChartDataItem[];
  category_distribution: ChartDataItem[];
  source_distribution: ChartDataItem[];
  edge_type_distribution: ChartDataItem[];
  // Computed metrics
  avg_edge_weight: number;
  avg_degree: number;
  unique_topics: number;
  unique_categories: number;
  unique_sources: number;
  // Extended info
  embedding?: {
    model: string;
    dimension: number;
  };
  config?: {
    neo4j_uri: string;
    auto_rebuild: boolean;
    log_level: string;
  };
}

export interface HealthStatus {
  status: "ok" | "error";
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
}

export interface SearchRequest {
  query_text: string;
  top_k?: number;
  candidate_k?: number;  // Optional, default 30
  // NOTE: No vector_weight or graph_weight - backend decides these adaptively
}

export interface VectorSearchRequest {
  query_text: string;
  top_k?: number;
  source_filter?: string;
  topic_filter?: string;
}

export interface GraphSearchRequest {
  start_id: string;
  depth?: number;
  max_nodes?: number;
}

// Graph traversal node from backend GraphSearchResponse
export interface GraphTraversalNode {
  node: Node;
  hop_distance: number;
  path_weight: number;
}

// Actual backend GraphSearchResponse structure
export interface GraphSearchResult {
  start_node: Node;
  traversed_nodes: GraphTraversalNode[];
  total_nodes: number;
  max_depth_reached: number;
}

// Vector search response from backend
export interface VectorSearchResult {
  node: Node;
  cosine_similarity: number;
  rank: number;
}

export interface VectorSearchResponse {
  query_text: string;
  results: VectorSearchResult[];
  total_results: number;
  search_time_ms: number;
}

export interface NeighborsResponse {
  nodes: Node[];
  edges: Edge[];
}

// Graph-only search response (no vectors involved)
export interface GraphOnlyResult {
  node: Node;
  match_type: string;      // "keyword" or "topic"
  hop_distance: number;
  edge_count: number;
}

export interface GraphOnlyResponse {
  query_text?: string;
  topic?: string;
  depth: number;
  results: GraphOnlyResult[];
  total_results: number;
  search_time_ms: number;
}
