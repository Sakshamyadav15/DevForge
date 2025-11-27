import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { searchApi, nodesApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
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
import type { SearchResult, Node, GraphSearchResult, VectorSearchResponse, GraphTraversalNode } from "@/types";

type SearchMode = "vector" | "graph" | "hybrid";

const Search = () => {
  const [searchMode, setSearchMode] = useState<SearchMode>("hybrid");
  
  // Vector search state
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  
  // Graph search state
  const [startNodeId, setStartNodeId] = useState("");
  const [traversalDepth, setTraversalDepth] = useState(2);
  const [relationshipType, setRelationshipType] = useState<string>("");
  
  // Hybrid search state
  const [vectorWeight, setVectorWeight] = useState([0.7]);
  const [graphWeight, setGraphWeight] = useState([0.3]);

  // Search results
  const [vectorResults, setVectorResults] = useState<VectorSearchResponse | null>(null);
  const [graphResults, setGraphResults] = useState<GraphSearchResult | null>(null);
  const [hybridResults, setHybridResults] = useState<{ results: SearchResult[] } | null>(null);

  // Fetch nodes for graph search dropdown
  const { data: nodesData } = useQuery({
    queryKey: ["nodes-for-search"],
    queryFn: () => nodesApi.getAll({ skip: 0, limit: 100 }),
  });

  // Vector search mutation
  const vectorSearch = useMutation({
    mutationFn: searchApi.vector,
    onSuccess: (data) => {
      setVectorResults(data);
      setGraphResults(null);
      setHybridResults(null);
    },
  });

  // Graph search mutation
  const graphSearch = useMutation({
    mutationFn: ({ startId, depth, relType }: { startId: string; depth: number; relType?: string }) =>
      searchApi.graph(startId, depth, relType || undefined),
    onSuccess: (data) => {
      setGraphResults(data);
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
      setGraphResults(null);
    },
  });

  const isPending = vectorSearch.isPending || graphSearch.isPending || hybridSearch.isPending;
  const error = vectorSearch.error || graphSearch.error || hybridSearch.error;

  const handleVectorSearch = () => {
    if (!query.trim()) return;
    vectorSearch.mutate({
      query_text: query,
      top_k: topK,
    });
  };

  const handleGraphSearch = () => {
    if (!startNodeId) return;
    graphSearch.mutate({
      startId: startNodeId,
      depth: traversalDepth,
      relType: relationshipType || undefined,
    });
  };

  const handleHybridSearch = () => {
    if (!query.trim()) return;
    hybridSearch.mutate({
      query_text: query,
      top_k: topK,
      vector_weight: vectorWeight[0],
      graph_weight: graphWeight[0],
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

  const renderGraphResults = (result: GraphSearchResult) => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold font-mono">
          Graph Traversal Results
        </h2>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="border-2">
            <Network className="h-3 w-3 mr-1" />
            Graph Search
          </Badge>
          <Badge variant="secondary" className="border-2">
            {result.total_nodes} nodes found
          </Badge>
          <Badge variant="secondary" className="border-2">
            Max depth: {result.max_depth_reached}
          </Badge>
        </div>
      </div>

      {/* Start Node */}
      <Card className="border-2 border-primary">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Start Node</CardTitle>
            <Badge className="bg-primary">Root</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <h3 className="font-bold">{result.start_node?.title || result.start_node?.id}</h3>
          <p className="text-xs text-muted-foreground font-mono">{result.start_node?.id}</p>
          <p className="text-sm mt-2 line-clamp-2">{result.start_node?.text}</p>
        </CardContent>
      </Card>

      {/* Traversed Nodes */}
      {result.traversed_nodes && result.traversed_nodes.length > 0 && (
        <div className="space-y-2">
          <h3 className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            Traversed Nodes ({result.traversed_nodes.length})
          </h3>
          <div className="grid gap-3">
            {result.traversed_nodes.map((traversalNode: GraphTraversalNode) => (
              <Card key={traversalNode.node.id} className="border-2">
                <CardContent className="py-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-bold">{traversalNode.node.title || traversalNode.node.id}</h4>
                      <p className="text-xs text-muted-foreground font-mono">{traversalNode.node.id}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="border-2">
                        Hop {traversalNode.hop_distance}
                      </Badge>
                      <Badge variant="secondary" className="border-2">
                        Weight: {traversalNode.path_weight.toFixed(2)}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm mt-2 line-clamp-2">{traversalNode.node.text}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {(!result.traversed_nodes || result.traversed_nodes.length === 0) && (
        <Card className="border-2">
          <CardContent className="py-8 text-center">
            <Network className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">No connected nodes found at depth {traversalDepth}</p>
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
        <Badge variant="outline" className="border-2">
          <Combine className="h-3 w-3 mr-1" />
          Hybrid Search
        </Badge>
      </div>

      {results.map((result, index) => (
        <Card key={result.id} className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-start justify-between mb-3">
              <div>
                <Badge variant="outline" className="border-2 font-mono mb-2">
                  #{index + 1}
                </Badge>
                <h3 className="font-bold text-lg mt-1">
                  {result.title || result.id}
                </h3>
                <p className="text-xs text-muted-foreground font-mono">
                  {result.id}
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold font-mono">
                  {result.score.toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Final Score
                </div>
              </div>
            </div>

            <p className="text-sm mb-4 leading-relaxed line-clamp-3">
              {result.text_snippet}
            </p>

            <div className="flex flex-wrap gap-4 text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Vector:</span>
                <Badge variant="secondary" className="border-2">
                  {result.vector_score.toFixed(3)}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Graph:</span>
                <Badge variant="secondary" className="border-2">
                  {result.graph_score.toFixed(3)}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Neighbors:</span>
                <Badge variant="outline" className="border-2">
                  {result.neighbors}
                </Badge>
              </div>
              {result.metadata?.topic && (
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">Topic:</span>
                  <Badge variant="outline" className="border-2">
                    {result.metadata.topic}
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

        {/* Graph Search Tab */}
        <TabsContent value="graph">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="font-mono text-sm uppercase tracking-wider">
                Graph Traversal Search
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm font-mono">Start Node</Label>
                  <Select value={startNodeId} onValueChange={setStartNodeId}>
                    <SelectTrigger className="border-2">
                      <SelectValue placeholder="Select a node" />
                    </SelectTrigger>
                    <SelectContent>
                      {nodesData?.items?.slice(0, 50).map((node: Node) => (
                        <SelectItem key={node.id} value={node.id}>
                          {(node.title || node.id).substring(0, 30)}...
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-mono">Traversal Depth</Label>
                  <Select value={String(traversalDepth)} onValueChange={(v) => setTraversalDepth(Number(v))}>
                    <SelectTrigger className="border-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 Hop</SelectItem>
                      <SelectItem value="2">2 Hops</SelectItem>
                      <SelectItem value="3">3 Hops</SelectItem>
                      <SelectItem value="4">4 Hops</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-mono">Relationship Type (Optional)</Label>
                  <Input
                    placeholder="e.g., RELATED_TO"
                    value={relationshipType}
                    onChange={(e) => setRelationshipType(e.target.value)}
                    className="border-2"
                  />
                </div>
              </div>
              <Button
                onClick={handleSearch}
                disabled={isPending || !startNodeId}
                className="border-2 w-full"
              >
                {isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Traversing Graph
                  </>
                ) : (
                  <>
                    <Network className="mr-2 h-4 w-4" />
                    Traverse Graph
                  </>
                )}
              </Button>
              <p className="text-xs text-muted-foreground">
                Explore connected nodes starting from a specific node in the knowledge graph
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Hybrid Search Tab */}
        <TabsContent value="hybrid">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="font-mono text-sm uppercase tracking-wider">
                Hybrid Search Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
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

              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label className="text-sm font-medium font-mono">
                    Vector Weight: {vectorWeight[0].toFixed(2)}
                  </Label>
                  <Slider
                    value={vectorWeight}
                    onValueChange={setVectorWeight}
                    min={0}
                    max={1}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Influence of semantic similarity
                  </p>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-medium font-mono">
                    Graph Weight: {graphWeight[0].toFixed(2)}
                  </Label>
                  <Slider
                    value={graphWeight}
                    onValueChange={setGraphWeight}
                    min={0}
                    max={1}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Influence of graph relationships
                  </p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Combine vector similarity with graph relationships for enhanced retrieval
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
      {graphResults && renderGraphResults(graphResults)}
      {hybridResults?.results && renderHybridResults(hybridResults.results)}

      {!vectorResults && !graphResults && !hybridResults && !isPending && (
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
