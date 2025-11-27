import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { edgesApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { useToast } from "@/hooks/use-toast";
import {
  ChevronLeft,
  ChevronRight,
  Plus,
  Trash2,
  Loader2,
  RefreshCw,
  ArrowRight,
} from "lucide-react";
import type { EdgeCreate } from "@/types";

const ITEMS_PER_PAGE = 15;
const ALL_TYPES_VALUE = "__all_edge_types__";

const EDGE_TYPES = [
  "RELATED_TO",
  "CITES",
  "SUPPORTS",
  "CONTRADICTS",
  "SIMILAR_TO",
  "DERIVED_FROM",
  "PART_OF",
];

const Edges = () => {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [page, setPage] = useState(0);
  const [typeFilter, setTypeFilter] = useState<string>(ALL_TYPES_VALUE);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  
  // Form state
  const [formData, setFormData] = useState<EdgeCreate>({
    source_id: "",
    target_id: "",
    type: "RELATED_TO",
    weight: 0.5,
  });

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["edges", page, typeFilter],
    queryFn: () =>
      edgesApi.list({
        limit: ITEMS_PER_PAGE,
        offset: page * ITEMS_PER_PAGE,
        type: typeFilter === ALL_TYPES_VALUE ? undefined : typeFilter,
      }),
    retry: 1,
  });

  const createMutation = useMutation({
    mutationFn: edgesApi.create,
    onSuccess: () => {
      toast({ title: "Success", description: "Edge created successfully" });
      setCreateDialogOpen(false);
      setFormData({ source_id: "", target_id: "", type: "RELATED_TO", weight: 0.5 });
      queryClient.invalidateQueries({ queryKey: ["edges"] });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to create edge",
        variant: "destructive",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: edgesApi.delete,
    onSuccess: () => {
      toast({ title: "Success", description: "Edge deleted successfully" });
      queryClient.invalidateQueries({ queryKey: ["edges"] });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to delete edge",
        variant: "destructive",
      });
    },
  });

  const handleCreate = () => {
    if (!formData.source_id.trim() || !formData.target_id.trim()) {
      toast({ title: "Error", description: "Source and Target IDs are required", variant: "destructive" });
      return;
    }
    createMutation.mutate(formData);
  };

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
          <h1 className="text-3xl font-bold tracking-tight">Edges</h1>
          <p className="text-muted-foreground mt-1">
            Manage relationships between nodes
          </p>
        </div>

        {/* Create Edge Dialog */}
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button className="border-2">
              <Plus className="mr-2 h-4 w-4" />
              Create Edge
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Create New Edge</DialogTitle>
              <DialogDescription>
                Create a relationship between two existing nodes.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="source">Source Node ID *</Label>
                <Input
                  id="source"
                  placeholder="Enter source node ID..."
                  value={formData.source_id}
                  onChange={(e) => setFormData({ ...formData, source_id: e.target.value })}
                  className="border-2 font-mono"
                />
              </div>
              <div className="flex items-center justify-center">
                <ArrowRight className="h-6 w-6 text-muted-foreground" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="target">Target Node ID *</Label>
                <Input
                  id="target"
                  placeholder="Enter target node ID..."
                  value={formData.target_id}
                  onChange={(e) => setFormData({ ...formData, target_id: e.target.value })}
                  className="border-2 font-mono"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="type">Relationship Type</Label>
                <Select
                  value={formData.type}
                  onValueChange={(v) => setFormData({ ...formData, type: v })}
                >
                  <SelectTrigger className="border-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {EDGE_TYPES.map((type) => (
                      <SelectItem key={type} value={type}>
                        {type}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Weight: {formData.weight?.toFixed(2)}</Label>
                <Slider
                  value={[formData.weight || 0.5]}
                  onValueChange={([v]) => setFormData({ ...formData, weight: v })}
                  min={0}
                  max={1}
                  step={0.05}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  0 = weak relationship, 1 = strong relationship
                </p>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateDialogOpen(false)} className="border-2">
                Cancel
              </Button>
              <Button onClick={handleCreate} disabled={createMutation.isPending} className="border-2">
                {createMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Create Edge
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
        <CardContent>
          <div className="flex justify-between items-center gap-4">
            <Select value={typeFilter} onValueChange={(v) => { setTypeFilter(v); setPage(0); }}>
              <SelectTrigger className="w-48 border-2">
                <SelectValue placeholder="All Types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_TYPES_VALUE}>All Types</SelectItem>
                {EDGE_TYPES.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <div className="flex items-center gap-4">
              <div className="text-sm text-muted-foreground font-mono">
                {data?.total.toLocaleString()} total edges
              </div>
              <Button variant="outline" onClick={() => refetch()} className="border-2">
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Table */}
      <Card className="border-2">
        <Table>
          <TableHeader>
            <TableRow className="border-b-2">
              <TableHead className="font-mono uppercase">Source</TableHead>
              <TableHead className="font-mono uppercase w-12"></TableHead>
              <TableHead className="font-mono uppercase">Target</TableHead>
              <TableHead className="font-mono uppercase">Type</TableHead>
              <TableHead className="font-mono uppercase">Weight</TableHead>
              <TableHead className="font-mono uppercase">Created</TableHead>
              <TableHead className="font-mono uppercase text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data?.items.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                  No edges found
                </TableCell>
              </TableRow>
            ) : (
              data?.items.map((edge) => (
                <TableRow key={edge.id} className="border-b-2">
                  <TableCell className="font-mono text-xs max-w-32 truncate">
                    {edge.source}
                  </TableCell>
                  <TableCell className="text-center">
                    <ArrowRight className="h-4 w-4 text-muted-foreground mx-auto" />
                  </TableCell>
                  <TableCell className="font-mono text-xs max-w-32 truncate">
                    {edge.target}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="border-2">
                      {edge.type}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full"
                          style={{ width: `${(edge.weight || 0) * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs">
                        {(edge.weight || 0).toFixed(2)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {new Date(edge.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell className="text-right">
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button variant="ghost" size="icon" title="Delete Edge">
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete Edge?</AlertDialogTitle>
                          <AlertDialogDescription>
                            This will permanently delete the relationship between these nodes.
                            This action cannot be undone.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel className="border-2">Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => deleteMutation.mutate(edge.id)}
                            className="bg-destructive text-destructive-foreground border-2"
                          >
                            Delete
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
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
    </div>
  );
};

export default Edges;
