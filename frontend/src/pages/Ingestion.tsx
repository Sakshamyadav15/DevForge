import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { statsApi, ingestionApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Plus, Check, AlertCircle, Loader2, Database, FileText } from "lucide-react";

const Ingestion = () => {
  const queryClient = useQueryClient();
  
  // Entity form state
  const [text, setText] = useState("");
  const [title, setTitle] = useState("");
  const [topic, setTopic] = useState("");
  const [category, setCategory] = useState("");
  
  // Document form state
  const [docContent, setDocContent] = useState("");
  const [docTitle, setDocTitle] = useState("");
  const [docTopic, setDocTopic] = useState("");
  const [docSource, setDocSource] = useState("");
  
  // Success/error messages
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const { data: stats, refetch: refetchStats } = useQuery({
    queryKey: ["stats"],
    queryFn: statsApi.get,
    retry: 1,
  });

  // Mutation for ingesting a single node/entity
  const ingestMutation = useMutation({
    mutationFn: (data: { text: string; metadata?: Record<string, any> }) =>
      ingestionApi.ingestNode(data),
    onSuccess: (response) => {
      setSuccessMessage(response.message);
      setErrorMessage(null);
      setText("");
      setTitle("");
      setTopic("");
      setCategory("");
      refetchStats();
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      setTimeout(() => setSuccessMessage(null), 5000);
    },
    onError: (error: any) => {
      setErrorMessage(
        error.response?.data?.detail || error.message || "Failed to ingest node"
      );
      setSuccessMessage(null);
    },
  });

  // Mutation for ingesting a document
  const documentMutation = useMutation({
    mutationFn: (data: {
      content: string;
      title?: string;
      topic?: string;
      source?: string;
    }) => ingestionApi.ingestDocument(data),
    onSuccess: (response) => {
      setSuccessMessage(
        `Document "${response.title}" ingested successfully! Created ${response.chunks_created} chunks with ${response.edges_created} relationships.`
      );
      setErrorMessage(null);
      setDocContent("");
      setDocTitle("");
      setDocTopic("");
      setDocSource("");
      refetchStats();
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      setTimeout(() => setSuccessMessage(null), 8000);
    },
    onError: (error: any) => {
      setErrorMessage(
        error.response?.data?.detail || error.message || "Failed to ingest document"
      );
      setSuccessMessage(null);
    },
  });

  const handleEntitySubmit = (e: React.FormEvent) => {
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

  const handleDocumentSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!docContent.trim() || docContent.trim().length < 10) {
      setErrorMessage("Document content must be at least 10 characters");
      return;
    }

    documentMutation.mutate({
      content: docContent.trim(),
      title: docTitle.trim() || undefined,
      topic: docTopic.trim() || undefined,
      source: docSource.trim() || undefined,
    });
  };

  const isLoading = ingestMutation.isPending || documentMutation.isPending;

  return (
    <div className="p-8 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Ingestion</h1>
        <p className="text-muted-foreground mt-1">
          Add entities or documents to the knowledge graph
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

      {/* Tabbed Ingestion Forms */}
      <Tabs defaultValue="entity" className="w-full">
        <TabsList className="grid w-full grid-cols-2 border-2">
          <TabsTrigger value="entity" className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            Single Entity
          </TabsTrigger>
          <TabsTrigger value="document" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Document (Unstructured)
          </TabsTrigger>
        </TabsList>

        {/* Single Entity Tab */}
        <TabsContent value="entity">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Plus className="h-5 w-5" />
                Add Single Entity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleEntitySubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="text" className="text-base font-medium">
                    Entity Text <span className="text-red-500">*</span>
                  </Label>
                  <Textarea
                    id="text"
                    placeholder="Enter the entity description or content..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="min-h-[150px] border-2 font-mono text-sm"
                    disabled={isLoading}
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="title">Title (optional)</Label>
                    <Input
                      id="title"
                      placeholder="Short title..."
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      className="border-2"
                      disabled={isLoading}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="topic">Topic (optional)</Label>
                    <Input
                      id="topic"
                      placeholder="e.g., technology"
                      value={topic}
                      onChange={(e) => setTopic(e.target.value)}
                      className="border-2"
                      disabled={isLoading}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="category">Category (optional)</Label>
                    <Input
                      id="category"
                      placeholder="e.g., concept"
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      className="border-2"
                      disabled={isLoading}
                    />
                  </div>
                </div>

                <div className="flex justify-end">
                  <Button
                    type="submit"
                    size="lg"
                    className="border-2 min-w-[200px]"
                    disabled={isLoading || !text.trim()}
                  >
                    {ingestMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Ingesting...
                      </>
                    ) : (
                      <>
                        <Plus className="mr-2 h-4 w-4" />
                        Add Entity
                      </>
                    )}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Document Tab */}
        <TabsContent value="document">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Ingest Unstructured Document
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleDocumentSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="docContent" className="text-base font-medium">
                    Document Content <span className="text-red-500">*</span>
                  </Label>
                  <Textarea
                    id="docContent"
                    placeholder="Paste your full document here... The system will automatically chunk it into connected entities."
                    value={docContent}
                    onChange={(e) => setDocContent(e.target.value)}
                    className="min-h-[250px] border-2 font-mono text-sm"
                    disabled={isLoading}
                  />
                  <p className="text-xs text-muted-foreground">
                    Large documents will be chunked into multiple connected nodes with sequential and semantic relationships.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="docTitle">Document Title (optional)</Label>
                    <Input
                      id="docTitle"
                      placeholder="Document name..."
                      value={docTitle}
                      onChange={(e) => setDocTitle(e.target.value)}
                      className="border-2"
                      disabled={isLoading}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="docTopic">Topic (optional)</Label>
                    <Input
                      id="docTopic"
                      placeholder="Auto-detected if empty"
                      value={docTopic}
                      onChange={(e) => setDocTopic(e.target.value)}
                      className="border-2"
                      disabled={isLoading}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="docSource">Source (optional)</Label>
                    <Input
                      id="docSource"
                      placeholder="e.g., research_paper"
                      value={docSource}
                      onChange={(e) => setDocSource(e.target.value)}
                      className="border-2"
                      disabled={isLoading}
                    />
                  </div>
                </div>

                <div className="flex justify-end">
                  <Button
                    type="submit"
                    size="lg"
                    className="border-2 min-w-[200px]"
                    disabled={isLoading || !docContent.trim() || docContent.trim().length < 10}
                  >
                    {documentMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <FileText className="mr-2 h-4 w-4" />
                        Ingest Document
                      </>
                    )}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Info */}
      <Alert className="border-2">
        <AlertDescription className="text-sm">
          <strong>How It Works:</strong>
          <ul className="mt-2 space-y-1 list-disc list-inside text-muted-foreground">
            <li><strong>Smart Chunking:</strong> Documents are automatically split into optimal chunks</li>
            <li><strong>Entity Extraction:</strong> Topics and entities are auto-detected from content</li>
            <li><strong>Relationship Building:</strong> Chunks are connected by sequence and semantic similarity</li>
            <li><strong>Full Persistence:</strong> Saved to Neo4j (graph), FAISS (vectors), and snapshot</li>
          </ul>
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default Ingestion;
