import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { statsApi, ingestionApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Plus, Check, AlertCircle, Loader2, Database } from "lucide-react";

const Ingestion = () => {
  const queryClient = useQueryClient();
  
  // Form state
  const [text, setText] = useState("");
  const [title, setTitle] = useState("");
  const [topic, setTopic] = useState("");
  const [category, setCategory] = useState("");
  
  // Success/error messages
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const { data: stats, refetch: refetchStats } = useQuery({
    queryKey: ["stats"],
    queryFn: statsApi.get,
    retry: 1,
  });

  // Mutation for ingesting a node
  const ingestMutation = useMutation({
    mutationFn: (data: { text: string; metadata?: Record<string, any> }) =>
      ingestionApi.ingestNode(data),
    onSuccess: (response) => {
      setSuccessMessage(response.message);
      setErrorMessage(null);
      // Clear form
      setText("");
      setTitle("");
      setTopic("");
      setCategory("");
      // Refresh stats
      refetchStats();
      // Invalidate nodes query so the Nodes page shows the new node
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
    },
    onError: (error: any) => {
      setErrorMessage(
        error.response?.data?.detail || error.message || "Failed to ingest node"
      );
      setSuccessMessage(null);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setErrorMessage("Text content is required");
      return;
    }

    const metadata: Record<string, any> = {};
    if (title.trim()) metadata.title = title.trim();
    if (topic.trim()) metadata.topic = topic.trim();
    if (category.trim()) metadata.category = category.trim();
    metadata.source = "web_ingestion";

    ingestMutation.mutate({
      text: text.trim(),
      metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
    });
  };

  return (
    <div className="p-8 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Ingestion</h1>
        <p className="text-muted-foreground mt-1">
          Add new entities to the knowledge graph
        </p>
      </div>

      {/* Current Stats */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider flex items-center gap-2">
            <Database className="h-4 w-4" />
            Current Database
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="p-4 border-2 border-border text-center">
              <div className="text-2xl font-bold font-mono">
                {stats?.nodes?.toLocaleString() || 0}
              </div>
              <div className="text-sm text-muted-foreground">Nodes</div>
            </div>
            <div className="p-4 border-2 border-border text-center">
              <div className="text-2xl font-bold font-mono">
                {stats?.edges?.toLocaleString() || 0}
              </div>
              <div className="text-sm text-muted-foreground">Edges</div>
            </div>
            <div className="p-4 border-2 border-border text-center">
              <div className="text-2xl font-bold font-mono">
                {stats?.vector_index_size?.toLocaleString() || 0}
              </div>
              <div className="text-sm text-muted-foreground">Vectors</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Success/Error Messages */}
      {successMessage && (
        <Alert className="border-2 border-green-500 bg-green-50 dark:bg-green-950">
          <Check className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-700 dark:text-green-300">
            {successMessage}
          </AlertDescription>
        </Alert>
      )}

      {errorMessage && (
        <Alert className="border-2 border-red-500 bg-red-50 dark:bg-red-950">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-700 dark:text-red-300">
            {errorMessage}
          </AlertDescription>
        </Alert>
      )}

      {/* Ingestion Form */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plus className="h-5 w-5" />
            Add New Entity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Text Content - Main field */}
            <div className="space-y-2">
              <Label htmlFor="text" className="text-base font-medium">
                Entity Text <span className="text-red-500">*</span>
              </Label>
              <Textarea
                id="text"
                placeholder="Enter the entity description or content... (e.g., 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.')"
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-[150px] border-2 font-mono text-sm"
                disabled={ingestMutation.isPending}
              />
              <p className="text-xs text-muted-foreground">
                This text will be embedded and stored in both the vector index and graph database.
              </p>
            </div>

            {/* Optional Metadata */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="title">Title (optional)</Label>
                <Input
                  id="title"
                  placeholder="Short title..."
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="border-2"
                  disabled={ingestMutation.isPending}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="topic">Topic (optional)</Label>
                <Input
                  id="topic"
                  placeholder="e.g., machine_learning"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  className="border-2"
                  disabled={ingestMutation.isPending}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="category">Category (optional)</Label>
                <Input
                  id="category"
                  placeholder="e.g., concept, definition"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="border-2"
                  disabled={ingestMutation.isPending}
                />
              </div>
            </div>

            {/* Submit Button */}
            <div className="flex justify-end">
              <Button
                type="submit"
                size="lg"
                className="border-2 min-w-[200px]"
                disabled={ingestMutation.isPending || !text.trim()}
              >
                {ingestMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Ingesting...
                  </>
                ) : (
                  <>
                    <Plus className="mr-2 h-4 w-4" />
                    Add to Graph
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Info */}
      <Alert className="border-2">
        <AlertDescription className="text-sm">
          <strong>What happens when you add an entity:</strong>
          <ul className="mt-2 space-y-1 list-disc list-inside text-muted-foreground">
            <li>The text is converted into a 384-dimensional embedding</li>
            <li>A new node is created in the Neo4j graph database</li>
            <li>The embedding is added to the FAISS vector index for similarity search</li>
            <li><strong>Relationships are automatically created</strong> with similar existing nodes</li>
            <li>The entity is persisted to snapshot.json for data backup</li>
          </ul>
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default Ingestion;
