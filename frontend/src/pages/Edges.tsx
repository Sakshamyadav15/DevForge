import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { edgesApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { ChevronLeft, ChevronRight } from "lucide-react";

const ITEMS_PER_PAGE = 15;

const Edges = () => {
  const [page, setPage] = useState(0);
  const [typeFilter, setTypeFilter] = useState<string>("");

  const { data, isLoading } = useQuery({
    queryKey: ["edges", page, typeFilter],
    queryFn: () =>
      edgesApi.getAll(page + 1, ITEMS_PER_PAGE),
    retry: 1,
  });

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
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Edges</h1>
        <p className="text-muted-foreground mt-1">
          Explore relationships between nodes
        </p>
      </div>

      <div className="flex justify-between items-center">
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          <SelectTrigger className="w-48 border-2">
            <SelectValue placeholder="All Types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="">All Types</SelectItem>
            <SelectItem value="citation">Citation</SelectItem>
            <SelectItem value="similarity">Similarity</SelectItem>
            <SelectItem value="coauthor">Co-author</SelectItem>
            <SelectItem value="topic">Topic</SelectItem>
            <SelectItem value="reference">Reference</SelectItem>
          </SelectContent>
        </Select>

        <div className="text-sm text-muted-foreground font-mono">
          {data?.total.toLocaleString()} total edges
        </div>
      </div>

      <Card className="border-2">
        <Table>
          <TableHeader>
            <TableRow className="border-b-2">
              <TableHead className="font-mono uppercase">Source</TableHead>
              <TableHead className="font-mono uppercase">Target</TableHead>
              <TableHead className="font-mono uppercase">Type</TableHead>
              <TableHead className="font-mono uppercase">Weight</TableHead>
              <TableHead className="font-mono uppercase">Topic</TableHead>
              <TableHead className="font-mono uppercase">Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data?.items.map((edge) => (
              <TableRow key={edge.id} className="border-b-2">
                <TableCell className="font-mono text-xs">{edge.source}</TableCell>
                <TableCell className="font-mono text-xs">{edge.target}</TableCell>
                <TableCell>
                  <Badge variant="outline" className="border-2">
                    {edge.type}
                  </Badge>
                </TableCell>
                <TableCell className="font-mono">
                  {edge.weight?.toFixed(3) || "N/A"}
                </TableCell>
                <TableCell>
                  <Badge variant="secondary" className="border-2">
                    {edge.topic}
                  </Badge>
                </TableCell>
                <TableCell className="font-mono text-xs">
                  {new Date(edge.created_at).toLocaleDateString()}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <div className="flex items-center justify-between p-4 border-t-2 border-border">
          <div className="text-sm text-muted-foreground">
            Showing {page * ITEMS_PER_PAGE + 1} to{" "}
            {Math.min((page + 1) * ITEMS_PER_PAGE, data?.total || 0)} of{" "}
            {data?.total} results
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
