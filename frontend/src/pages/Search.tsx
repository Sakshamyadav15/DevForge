import { useState, useMemo } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { searchApi, statsApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search as SearchIcon,
  Loader2,
  Database,
  Network,
  Combine,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import type { SearchResult, Node, VectorSearchResponse } from "@/types";

type SearchMode = "vector" | "graph" | "hybrid";

// Graph-only search result type
interface GraphOnlyResult {
  id: string;
  text_snippet: string;
  topic: string;
  hop_distance: number;
  degree: number;
  graph_score: number;
  is_direct_match: boolean;
}

interface GraphOnlyResponse {
  search_type: string;
  query_text: string | null;
  topic_filter: string | null;
  depth: number;
  direct_matches: number;
  traversed_found: number;
  total_results: number;
  results: GraphOnlyResult[];
}

const Search = () => {
  const [searchMode, setSearchMode] = useState<SearchMode>("hybrid");
  
  // Common search state
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  
  // Graph search state
  const [graphTopic, setGraphTopic] = useState<string>("");
  const [traversalDepth, setTraversalDepth] = useState(2);

  // Search results
  const [vectorResults, setVectorResults] = useState<VectorSearchResponse | null>(null);
  const [graphOnlyResults, setGraphOnlyResults] = useState<GraphOnlyResponse | null>(null);
  const [hybridResults, setHybridResults] = useState<{ results: SearchResult[] } | null>(null);

  // Fetch stats to get available topics
  const { data: statsData } = useQuery({
    queryKey: ["stats-for-search"],
    queryFn: () => statsApi.get(),
  });

  // Get topics from stats
  const availableTopics = useMemo(() => {
    if (!statsData?.topic_distribution) return [];
    return statsData.topic_distribution.map((item: { name: string }) => item.name);
  }, [statsData]);

  // Vector search mutation
  const vectorSearch = useMutation({
    mutationFn: searchApi.vector,
    onSuccess: (data) => {
      setVectorResults(data);
      setGraphOnlyResults(null);
      setHybridResults(null);
    },
  });

  // Graph-only search mutation (NO vectors!)
  const graphOnlySearch = useMutation({
    mutationFn: (params: { query_text?: string; topic?: string; depth?: number; top_k?: number }) =>
      searchApi.graphOnly(params),
    onSuccess: (data) => {
      setGraphOnlyResults(data);
      setVectorResults(null);
      setHybridResults(null);
    },
  });

  // Hybrid search mutation
  const hybridSearch = useMutation({
    mutationFn: searchApi.hybrid,
    onSuccess: (data) => {
      setHybridResults(data);
      setVectorResults(null);
      setGraphOnlyResults(null);
    },
  });

  const isPending = vectorSearch.isPending || graphOnlySearch.isPending || hybridSearch.isPending;
  const error = vectorSearch.error || graphOnlySearch.error || hybridSearch.error;

  const handleVectorSearch = () => {
    if (!query.trim()) return;
    vectorSearch.mutate({
      query_text: query,
      top_k: topK,
    });
  };

  const handleGraphSearch = () => {
    // Graph search can work with keyword, topic, or both
    graphOnlySearch.mutate({
      query_text: query.trim() || undefined,
      topic: graphTopic || undefined,
      depth: traversalDepth,
      top_k: topK,
    });
  };

  const handleHybridSearch = () => {
    if (!query.trim()) return;
    // Backend decides weights adaptively based on query intent
    hybridSearch.mutate({
      query_text: query,
      top_k: topK,
      candidate_k: 30,  // Fetch more candidates for better re-ranking
    });
  };

  const handleSearch = () => {
    switch (searchMode) {
      case "vector":
        handleVectorSearch();
        break;
      case "graph":
        handleGraphSearch();
        break;
      case "hybrid":
        handleHybridSearch();
        break;
    }
  };

  const renderVectorResults = (response: VectorSearchResponse) => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold font-mono">
          {response.total_results} Results
        </h2>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="border-2">
            <Database className="h-3 w-3 mr-1" />
            Vector Search
          </Badge>
          <Badge variant="secondary" className="border-2">
            {response.search_time_ms.toFixed(1)}ms
          </Badge>
        </div>
      </div>

      {response.results.map((result) => (
        <Card key={result.node.id} className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-start justify-between mb-3">
              <div>
                <Badge variant="outline" className="border-2 font-mono mb-2">
                  #{result.rank}
                </Badge>
                <h3 className="font-bold text-lg mt-1">
                  {result.node.title || result.node.id}
                </h3>
                <p className="text-xs text-muted-foreground font-mono">
                  {result.node.id}
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold font-mono">
                  {result.cosine_similarity.toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Similarity Score
                </div>
              </div>
            </div>

            <p className="text-sm mb-4 leading-relaxed line-clamp-3">
              {result.node.text}
            </p>

            {result.node.metadata?.topic && (
              <Badge variant="secondary" className="border-2">
                {result.node.metadata.topic}
              </Badge>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );

  // Graph-only search results renderer
  const renderGraphOnlyResults = (response: GraphOnlyResponse) => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold font-mono">
          Graph-Only Results
        </h2>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="border-2">
            <Network className="h-3 w-3 mr-1" />
            Graph Search (No Vectors)
          </Badge>
          <Badge variant="secondary" className="border-2">
            {response.total_results} results
          </Badge>
        </div>
      </div>

      {/* Search info */}
      <div className="flex flex-wrap gap-2 text-sm">
        {response.query_text && (
          <Badge variant="outline">Keyword: "{response.query_text}"</Badge>
        )}
        {response.topic_filter && (
          <Badge variant="outline">Topic: {response.topic_filter}</Badge>
        )}
        <Badge variant="secondary">Depth: {response.depth}</Badge>
        <Badge variant="secondary">Direct matches: {response.direct_matches}</Badge>
        <Badge variant="secondary">Traversed: {response.traversed_found}</Badge>
      </div>

      {/* Results */}
      {response.results.map((result, index) => (
        <Card key={result.id} className={`border-2 ${result.is_direct_match ? 'border-primary' : ''}`}>
          <CardContent className="pt-6">
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline" className="border-2 font-mono">
                    #{index + 1}
                  </Badge>
                  {result.is_direct_match && (
                    <Badge className="bg-primary">Direct Match</Badge>
                  )}
                  {!result.is_direct_match && (
                    <Badge variant="secondary">Hop {result.hop_distance}</Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground font-mono">
                  {result.id}
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold font-mono">
                  {result.graph_score.toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Graph Score
                </div>
              </div>
            </div>

            <p className="text-sm mb-4 leading-relaxed">
              {result.text_snippet}
            </p>

            <div className="flex flex-wrap gap-2">
              {result.topic && (
                <Badge variant="secondary">{result.topic}</Badge>
              )}
              <Badge variant="outline">Connections: {result.degree}</Badge>
            </div>
          </CardContent>
        </Card>
      ))}

      {response.results.length === 0 && (
        <Card className="border-2">
          <CardContent className="py-8 text-center">
            <Network className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">No nodes found. Try different keywords or topic.</p>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderHybridResults = (results: SearchResult[]) => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold font-mono">
          {results.length} Results
        </h2>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="border-2">
            <Combine className="h-3 w-3 mr-1" />
            Hybrid Search
          </Badge>
          <Badge variant="secondary" className="border-2 text-xs">
            Adaptive Weights
          </Badge>
        </div>
      </div>

      {results.map((result, index) => (
        <Card key={result.id} className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex gap-2 mb-2">
                  <Badge variant="outline" className="border-2 font-mono">
                    #{result.rank || index + 1}
                  </Badge>
                  {result.vector_only_rank && result.vector_only_rank !== result.rank && (
                    <Badge variant="secondary" className="text-xs">
                      Vector-only: #{result.vector_only_rank}
                    </Badge>
                  )}
                </div>
                <h3 className="font-bold text-lg mt-1">
                  {result.metadata?.title || result.title || result.id}
                </h3>
                <p className="text-xs text-muted-foreground font-mono">
                  {result.id}
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold font-mono">
                  {(result.final_score ?? result.score ?? 0).toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Final Score
                </div>
              </div>
            </div>

            <p className="text-sm mb-4 leading-relaxed line-clamp-3">
              {result.text_snippet || result.text}
            </p>

            <div className="flex flex-wrap gap-4 text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Cosine:</span>
                <Badge variant="secondary" className="border-2">
                  {(result.cosine_sim ?? result.vector_score ?? 0).toFixed(3)}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Graph:</span>
                <Badge variant="secondary" className="border-2">
                  {(result.graph_score ?? 0).toFixed(3)}
                </Badge>
              </div>
              {(result.degree ?? result.neighbors) !== undefined && (
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">Degree:</span>
                  <Badge variant="outline" className="border-2">
                    {result.degree ?? result.neighbors}
                  </Badge>
                </div>
              )}
              {(result.topic || result.metadata?.topic) && (
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">Topic:</span>
                  <Badge variant="outline" className="border-2">
                    {result.topic || result.metadata?.topic}
                  </Badge>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  return (
    <div className="p-8 space-y-6 max-w-6xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Search</h1>
        <p className="text-muted-foreground mt-1">
          Search using Vector similarity, Graph traversal, or Hybrid approach
        </p>
      </div>

      <Tabs value={searchMode} onValueChange={(v) => setSearchMode(v as SearchMode)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="vector" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Vector
          </TabsTrigger>
          <TabsTrigger value="graph" className="flex items-center gap-2">
            <Network className="h-4 w-4" />
            Graph
          </TabsTrigger>
          <TabsTrigger value="hybrid" className="flex items-center gap-2">
            <Combine className="h-4 w-4" />
            Hybrid
          </TabsTrigger>
        </TabsList>

        {/* Vector Search Tab */}
        <TabsContent value="vector">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="font-mono text-sm uppercase tracking-wider">
                Vector Similarity Search
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <Input
                    placeholder="Enter search query..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    className="border-2"
                  />
                </div>
                <div className="w-32">
                  <Select value={String(topK)} onValueChange={(v) => setTopK(Number(v))}>
                    <SelectTrigger className="border-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5">Top 5</SelectItem>
                      <SelectItem value="10">Top 10</SelectItem>
                      <SelectItem value="20">Top 20</SelectItem>
                      <SelectItem value="50">Top 50</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  onClick={handleSearch}
                  disabled={isPending || !query.trim()}
                  className="border-2 min-w-32"
                >
                  {isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Searching
                    </>
                  ) : (
                    <>
                      <SearchIcon className="mr-2 h-4 w-4" />
                      Search
                    </>
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Find documents similar to your query using vector embeddings (FAISS)
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Graph Search Tab - Simple keyword + topic search */}
        <TabsContent value="graph">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="font-mono text-sm uppercase tracking-wider">
                Graph Search (Keyword + Topic)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Simple keyword search */}
              <div className="space-y-2">
                <Label className="text-sm font-mono">Search Keywords</Label>
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter keywords to search in graph (e.g., AI, machine learning)..."
                  className="border-2"
                />
              </div>

              {/* Topic filter dropdown */}
              <div className="space-y-2">
                <Label className="text-sm font-mono">Filter by Topic (Optional)</Label>
                <Select value={graphTopic || "ALL"} onValueChange={(v) => setGraphTopic(v === "ALL" ? "" : v)}>
                  <SelectTrigger className="border-2">
                    <SelectValue placeholder="Select a topic" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL">All Topics</SelectItem>
                    {availableTopics.map((topic: string) => (
                      <SelectItem key={topic} value={topic}>
                        {topic}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm font-mono">Traversal Depth</Label>
                  <Select value={String(traversalDepth)} onValueChange={(v) => setTraversalDepth(Number(v))}>
                    <SelectTrigger className="border-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 Hop (Direct connections)</SelectItem>
                      <SelectItem value="2">2 Hops</SelectItem>
                      <SelectItem value="3">3 Hops</SelectItem>
                      <SelectItem value="4">4 Hops (Deep exploration)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-mono">Max Results</Label>
                  <Select value={String(topK)} onValueChange={(v) => setTopK(Number(v))}>
                    <SelectTrigger className="border-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5">Top 5</SelectItem>
                      <SelectItem value="10">Top 10</SelectItem>
                      <SelectItem value="20">Top 20</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Button
                onClick={handleSearch}
                disabled={graphOnlySearch.isPending || !query.trim()}
                className="border-2 w-full"
              >
                {graphOnlySearch.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Searching Graph
                  </>
                ) : (
                  <>
                    <Network className="mr-2 h-4 w-4" />
                    Search Graph
                  </>
                )}
              </Button>
              <p className="text-xs text-muted-foreground">
                Search documents by keywords and optionally filter by topic, then explore connections in the graph
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Hybrid Search Tab */}
        <TabsContent value="hybrid">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="font-mono text-sm uppercase tracking-wider">
                Hybrid Search (Vector + Graph)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <Input
                    placeholder="Enter search query (e.g., 'AI in healthcare' or 'connection between finance and AI')..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    className="border-2"
                  />
                </div>
                <div className="w-32">
                  <Select value={String(topK)} onValueChange={(v) => setTopK(Number(v))}>
                    <SelectTrigger className="border-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5">Top 5</SelectItem>
                      <SelectItem value="10">Top 10</SelectItem>
                      <SelectItem value="20">Top 20</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  onClick={handleSearch}
                  disabled={isPending || !query.trim()}
                  className="border-2 min-w-32"
                >
                  {isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Searching
                    </>
                  ) : (
                    <>
                      <Combine className="mr-2 h-4 w-4" />
                      Hybrid Search
                    </>
                  )}
                </Button>
              </div>

              <div className="bg-muted/50 rounded-lg p-3 text-sm">
                <p className="font-medium mb-1">🧠 Adaptive Weights</p>
                <p className="text-muted-foreground text-xs">
                  The backend automatically adjusts vector/graph weights based on your query:
                </p>
                <ul className="text-xs text-muted-foreground mt-1 space-y-0.5">
                  <li>• <strong>Relationship queries</strong> ("between", "connected", "path") → More graph weight (60%)</li>
                  <li>• <strong>Semantic queries</strong> (general text) → More vector weight (70%)</li>
                </ul>
              </div>
              <p className="text-xs text-muted-foreground">
                Combines FAISS vector similarity with Neo4j graph structure for best results
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {error && (
        <Alert className="border-2 border-red-500">
          <AlertDescription>
            Search failed: {(error as Error).message || "Please check your backend connection."}
          </AlertDescription>
        </Alert>
      )}

      {/* Results Section */}
      {vectorResults && renderVectorResults(vectorResults)}
      {graphOnlyResults && renderGraphOnlyResults(graphOnlyResults)}
      {hybridResults?.results && renderHybridResults(hybridResults.results)}

      {!vectorResults && !graphOnlyResults && !hybridResults && !isPending && (
        <Card className="border-2">
          <CardContent className="py-12 text-center">
            <SearchIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">
              Configure your search and click the search button to see results
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Search;
