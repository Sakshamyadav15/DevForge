import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { searchApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Search as SearchIcon, Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import type { SearchResult } from "@/types";

const Search = () => {
  const [query, setQuery] = useState("");
  const [vectorWeight, setVectorWeight] = useState([0.7]);
  const [graphWeight, setGraphWeight] = useState([0.3]);

  const {
    mutate: performSearch,
    data: results,
    isPending,
    error,
  } = useMutation({
    mutationFn: searchApi.hybrid,
  });

  const handleSearch = () => {
    if (!query.trim()) return;

    performSearch({
      query_text: query,
      top_k: 10,
      vector_weight: vectorWeight[0],
      graph_weight: graphWeight[0],
    });
  };

  const highlightText = (text: string, query: string) => {
    const parts = text.split(new RegExp(`(${query})`, "gi"));
    return parts.map((part, i) =>
      part.toLowerCase() === query.toLowerCase() ? (
        <mark key={i} className="bg-accent font-bold">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <div className="p-8 space-y-6 max-w-6xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Hybrid Search</h1>
        <p className="text-muted-foreground mt-1">
          Search using vector similarity + graph relationships
        </p>
      </div>

      <Card className="border-2">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider">
            Search Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex gap-4">
            <Input
              placeholder="Enter search query..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="border-2"
            />
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
              <label className="text-sm font-medium font-mono">
                Vector Weight: {vectorWeight[0].toFixed(2)}
              </label>
              <Slider
                value={vectorWeight}
                onValueChange={setVectorWeight}
                min={0}
                max={1}
                step={0.1}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium font-mono">
                Graph Weight: {graphWeight[0].toFixed(2)}
              </label>
              <Slider
                value={graphWeight}
                onValueChange={setGraphWeight}
                min={0}
                max={1}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert className="border-2 border-red-500">
          <AlertDescription>
            Search failed. Please check your backend connection.
          </AlertDescription>
        </Alert>
      )}

      {results && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold font-mono">
              {results.results?.length || 0} Results
            </h2>
          </div>

          {results.results?.map((result: SearchResult) => (
            <Card key={result.node.id} className="border-2">
              <CardContent className="pt-6">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <Badge variant="outline" className="border-2 font-mono mb-2">
                      Rank #{result.rank}
                    </Badge>
                    <span className="ml-2 text-xs text-muted-foreground">
                      (Vector-only rank: #{result.vector_only_rank})
                    </span>
                    <h3 className="font-bold text-lg mt-1">
                      {result.node.id}
                    </h3>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold font-mono">
                      {result.final_score.toFixed(3)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Final Score
                    </div>
                  </div>
                </div>

                <p className="text-sm mb-4 leading-relaxed line-clamp-3">
                  {result.node.text?.substring(0, 300)}
                  {(result.node.text?.length || 0) > 300 ? "..." : ""}
                </p>

                <div className="flex flex-wrap gap-4 text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">Cosine:</span>
                    <Badge variant="secondary" className="border-2">
                      {result.cosine_similarity.toFixed(3)}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">Graph:</span>
                    <Badge variant="secondary" className="border-2">
                      {result.graph_score.toFixed(3)}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">Degree:</span>
                    <Badge variant="outline" className="border-2">
                      {result.degree}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">Avg Weight:</span>
                    <Badge variant="outline" className="border-2">
                      {result.avg_edge_weight.toFixed(2)}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {!results && !isPending && (
        <Card className="border-2">
          <CardContent className="py-12 text-center">
            <SearchIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">
              Enter a query and click search to see results
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Search;
