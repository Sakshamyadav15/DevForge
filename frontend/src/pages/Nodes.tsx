import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { nodesApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import {
  Search,
  ChevronLeft,
  ChevronRight,
  Plus,
  Eye,
  Pencil,
  Trash2,
  Loader2,
  Network,
  RefreshCw,
} from "lucide-react";
import type { Node, NodeCreate, NodeUpdate, NeighborsResponse } from "@/types";

const ITEMS_PER_PAGE = 10;

const Nodes = () => {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // State
  const [page, setPage] = useState(0);
  const [topicFilter, setTopicFilter] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState("");
  
  // Dialog states
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [detailSheetOpen, setDetailSheetOpen] = useState(false);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [neighbors, setNeighbors] = useState<NeighborsResponse | null>(null);
  
  // Form state
  const [formData, setFormData] = useState<NodeCreate>({
    text: "",
    metadata: { title: "", topic: "", category: "", source: "" },
  });
  const [editFormData, setEditFormData] = useState<NodeUpdate>({
    text: "",
    metadata: {},
    regenerate_embedding: false,
  });

  // Queries
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["nodes", page, topicFilter],
    queryFn: () =>
      nodesApi.list({
        limit: ITEMS_PER_PAGE,
        offset: page * ITEMS_PER_PAGE,
        topic: topicFilter || undefined,
      }),
    retry: 1,
  });

  // Mutations
  const createMutation = useMutation({
    mutationFn: nodesApi.create,
    onSuccess: () => {
      toast({ title: "Success", description: "Node created successfully" });
      setCreateDialogOpen(false);
      setFormData({ text: "", metadata: { title: "", topic: "", category: "", source: "" } });
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to create node",
        variant: "destructive",
      });
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: NodeUpdate }) =>
      nodesApi.update(id, data),
    onSuccess: () => {
      toast({ title: "Success", description: "Node updated successfully" });
      setEditDialogOpen(false);
      setSelectedNode(null);
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to update node",
        variant: "destructive",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: nodesApi.delete,
    onSuccess: () => {
      toast({ title: "Success", description: "Node deleted successfully" });
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to delete node",
        variant: "destructive",
      });
    },
  });

  // Handlers
  const handleViewNode = async (node: Node) => {
    setSelectedNode(node);
    setDetailSheetOpen(true);
    try {
      const neighborsData = await nodesApi.getNeighbors(node.id);
      setNeighbors(neighborsData);
    } catch {
      setNeighbors(null);
    }
  };

  const handleEditNode = (node: Node) => {
    setSelectedNode(node);
    setEditFormData({
      text: node.text || "",
      metadata: node.metadata || {},
      regenerate_embedding: false,
    });
    setEditDialogOpen(true);
  };

  const handleCreate = () => {
    if (!formData.text.trim()) {
      toast({ title: "Error", description: "Text is required", variant: "destructive" });
      return;
    }
    createMutation.mutate(formData);
  };

  const handleUpdate = () => {
    if (!selectedNode) return;
    updateMutation.mutate({ id: selectedNode.id, data: editFormData });
  };

  const filteredItems = data?.items.filter((node) =>
    searchTerm
      ? node.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        node.text?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        node.id.toLowerCase().includes(searchTerm.toLowerCase())
      : true
  );

  const totalPages = Math.ceil((data?.total || 0) / ITEMS_PER_PAGE);

  if (isLoading) {
    return (
      <div className="p-8 space-y-6">
        <Skeleton className="h-12 w-64" />
        <Skeleton className="h-96" />
      </div>
    );
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Nodes</h1>
          <p className="text-muted-foreground mt-1">
            Manage document nodes in the knowledge graph
          </p>
        </div>
        
        {/* Create Node Dialog */}
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button className="border-2">
              <Plus className="mr-2 h-4 w-4" />
              Create Node
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Create New Node</DialogTitle>
              <DialogDescription>
                Add a new document node to the knowledge graph. An embedding will be automatically generated.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="title">Title</Label>
                <Input
                  id="title"
                  placeholder="Enter node title..."
                  value={formData.metadata?.title || ""}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      metadata: { ...formData.metadata, title: e.target.value },
                    })
                  }
                  className="border-2"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="text">Text Content *</Label>
                <Textarea
                  id="text"
                  placeholder="Enter the main text content..."
                  value={formData.text}
                  onChange={(e) => setFormData({ ...formData, text: e.target.value })}
                  className="border-2 min-h-32"
                />
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="topic">Topic</Label>
                  <Input
                    id="topic"
                    placeholder="e.g., arxiv"
                    value={formData.metadata?.topic || ""}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        metadata: { ...formData.metadata, topic: e.target.value },
                      })
                    }
                    className="border-2"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="category">Category</Label>
                  <Input
                    id="category"
                    placeholder="e.g., research"
                    value={formData.metadata?.category || ""}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        metadata: { ...formData.metadata, category: e.target.value },
                      })
                    }
                    className="border-2"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="source">Source</Label>
                  <Input
                    id="source"
                    placeholder="e.g., manual"
                    value={formData.metadata?.source || ""}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        metadata: { ...formData.metadata, source: e.target.value },
                      })
                    }
                    className="border-2"
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateDialogOpen(false)} className="border-2">
                Cancel
              </Button>
              <Button onClick={handleCreate} disabled={createMutation.isPending} className="border-2">
                {createMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Create Node
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Filters */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider">
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by ID, title, or text..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 border-2"
              />
            </div>
            <Select value={topicFilter} onValueChange={(v) => { setTopicFilter(v); setPage(0); }}>
              <SelectTrigger className="w-48 border-2">
                <SelectValue placeholder="All Topics" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Topics</SelectItem>
                <SelectItem value="arxiv">ArXiv</SelectItem>
                <SelectItem value="pubmed">PubMed</SelectItem>
                <SelectItem value="research">Research</SelectItem>
                <SelectItem value="technical">Technical</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" onClick={() => refetch()} className="border-2">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Table */}
      <Card className="border-2">
        <Table>
          <TableHeader>
            <TableRow className="border-b-2">
              <TableHead className="font-mono uppercase">ID</TableHead>
              <TableHead className="font-mono uppercase">Title</TableHead>
              <TableHead className="font-mono uppercase">Topic</TableHead>
              <TableHead className="font-mono uppercase">Category</TableHead>
              <TableHead className="font-mono uppercase">Created</TableHead>
              <TableHead className="font-mono uppercase text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredItems?.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                  No nodes found
                </TableCell>
              </TableRow>
            ) : (
              filteredItems?.map((node) => (
                <TableRow key={node.id} className="border-b-2">
                  <TableCell className="font-mono text-xs max-w-24 truncate">
                    {node.id.substring(0, 12)}...
                  </TableCell>
                  <TableCell className="max-w-md truncate font-medium">
                    {node.title || node.metadata?.title || "Untitled"}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="border-2">
                      {node.topic || node.metadata?.topic || "—"}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary" className="border-2">
                      {node.category || node.metadata?.category || "—"}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {new Date(node.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleViewNode(node)}
                        title="View Details"
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleEditNode(node)}
                        title="Edit Node"
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="ghost" size="icon" title="Delete Node">
                            <Trash2 className="h-4 w-4 text-destructive" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete Node?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This will permanently delete the node and all its associated edges. 
                              This action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel className="border-2">Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => deleteMutation.mutate(node.id)}
                              className="bg-destructive text-destructive-foreground border-2"
                            >
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>

        {/* Pagination */}
        <div className="flex items-center justify-between p-4 border-t-2 border-border">
          <div className="text-sm text-muted-foreground">
            {data?.total ? (
              <>
                Showing {page * ITEMS_PER_PAGE + 1} to{" "}
                {Math.min((page + 1) * ITEMS_PER_PAGE, data.total)} of {data.total} results
              </>
            ) : (
              "No results"
            )}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(page - 1)}
              disabled={page === 0}
              className="border-2"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(page + 1)}
              disabled={page >= totalPages - 1}
              className="border-2"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Edit Node</DialogTitle>
            <DialogDescription>
              Update node content and metadata. Optionally regenerate the embedding.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Node ID</Label>
              <Input value={selectedNode?.id || ""} disabled className="border-2 bg-muted" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-text">Text Content</Label>
              <Textarea
                id="edit-text"
                value={editFormData.text || ""}
                onChange={(e) => setEditFormData({ ...editFormData, text: e.target.value })}
                className="border-2 min-h-32"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="edit-title">Title</Label>
                <Input
                  id="edit-title"
                  value={editFormData.metadata?.title || ""}
                  onChange={(e) =>
                    setEditFormData({
                      ...editFormData,
                      metadata: { ...editFormData.metadata, title: e.target.value },
                    })
                  }
                  className="border-2"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="edit-topic">Topic</Label>
                <Input
                  id="edit-topic"
                  value={editFormData.metadata?.topic || ""}
                  onChange={(e) =>
                    setEditFormData({
                      ...editFormData,
                      metadata: { ...editFormData.metadata, topic: e.target.value },
                    })
                  }
                  className="border-2"
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="regenerate"
                checked={editFormData.regenerate_embedding}
                onChange={(e) =>
                  setEditFormData({ ...editFormData, regenerate_embedding: e.target.checked })
                }
                className="rounded border-2"
              />
              <Label htmlFor="regenerate" className="text-sm">
                Regenerate embedding (recommended if text changed)
              </Label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)} className="border-2">
              Cancel
            </Button>
            <Button onClick={handleUpdate} disabled={updateMutation.isPending} className="border-2">
              {updateMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Detail Sheet */}
      <Sheet open={detailSheetOpen} onOpenChange={setDetailSheetOpen}>
        <SheetContent className="w-[600px] sm:max-w-[600px] overflow-y-auto">
          <SheetHeader>
            <SheetTitle className="font-mono text-sm">{selectedNode?.id}</SheetTitle>
            <SheetDescription>
              {selectedNode?.title || selectedNode?.metadata?.title || "Node Details"}
            </SheetDescription>
          </SheetHeader>
          
          {selectedNode && (
            <div className="space-y-6 mt-6">
              {/* Metadata */}
              <div className="space-y-2">
                <h4 className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
                  Metadata
                </h4>
                <div className="flex flex-wrap gap-2">
                  {(selectedNode.topic || selectedNode.metadata?.topic) && (
                    <Badge variant="outline" className="border-2">
                      Topic: {selectedNode.topic || selectedNode.metadata?.topic}
                    </Badge>
                  )}
                  {(selectedNode.category || selectedNode.metadata?.category) && (
                    <Badge variant="secondary" className="border-2">
                      Category: {selectedNode.category || selectedNode.metadata?.category}
                    </Badge>
                  )}
                  {selectedNode.metadata?.source && (
                    <Badge variant="outline" className="border-2">
                      Source: {selectedNode.metadata.source}
                    </Badge>
                  )}
                </div>
              </div>

              {/* Text Content */}
              <div className="space-y-2">
                <h4 className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
                  Text Content
                </h4>
                <div className="p-4 bg-muted rounded-lg border-2 border-border text-sm whitespace-pre-wrap max-h-64 overflow-y-auto">
                  {selectedNode.text || "No text content"}
                </div>
              </div>

              {/* Neighbors */}
              <div className="space-y-2">
                <h4 className="font-mono text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                  <Network className="h-4 w-4" />
                  Neighbors ({neighbors?.nodes?.length || 0})
                </h4>
                {neighbors && neighbors.nodes && neighbors.nodes.length > 0 ? (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {neighbors.nodes.slice(0, 10).map((n) => (
                      <div
                        key={n.id}
                        className="p-3 border-2 border-border rounded-lg hover:bg-muted/50 cursor-pointer"
                        onClick={() => {
                          setSelectedNode(n);
                          nodesApi.getNeighbors(n.id).then(setNeighbors).catch(() => setNeighbors(null));
                        }}
                      >
                        <div className="font-mono text-xs text-muted-foreground">{n.id}</div>
                        <div className="font-medium truncate">
                          {n.title || n.metadata?.title || "Untitled"}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No neighbors found</p>
                )}
              </div>

              {/* Timestamps */}
              <div className="space-y-2">
                <h4 className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
                  Timestamps
                </h4>
                <div className="text-sm">
                  <span className="text-muted-foreground">Created:</span>{" "}
                  {new Date(selectedNode.created_at).toLocaleString()}
                </div>
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </div>
  );
};

export default Nodes;
