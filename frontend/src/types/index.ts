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
  id: string;
  title?: string;
  text_snippet: string;
  score: number;
  vector_score: number;
  graph_score: number;
  neighbors: number;
  metadata: Record<string, any>;
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
  query: string;
  vector_weight?: number;
  graph_weight?: number;
  filters?: {
    topic?: string;
    category?: string;
  };
}

export interface NeighborsResponse {
  nodes: Node[];
  edges: Edge[];
}
