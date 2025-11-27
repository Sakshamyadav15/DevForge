export interface Node {
  id: string;
  title?: string;
  text?: string;
  topic?: string;
  category?: string;
  created_at: string;
  metadata?: Record<string, any>;
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

export interface SearchResult {
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

export interface Stats {
  nodes: number;
  edges: number;
  snapshot_last_updated: string;
  vector_index_size?: number;
  graph_degree_avg?: number;
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
  vector_weight?: number;
  graph_weight?: number;
  source_filter?: string;
  topic_filter?: string;
}

export interface NeighborsResponse {
  nodes: Node[];
  edges: Edge[];
}
