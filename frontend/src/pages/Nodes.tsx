import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { nodesApi } from "@/lib/api";
import { mockNodes } from "@/lib/mockData";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";

const ITEMS_PER_PAGE = 10;

const Nodes = () => {
  const [page, setPage] = useState(0);
  const [topicFilter, setTopicFilter] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["nodes", page, topicFilter],
    queryFn: () =>
      nodesApi.list({
        limit: ITEMS_PER_PAGE,
        offset: page * ITEMS_PER_PAGE,
        topic: topicFilter || undefined,
      }),
    placeholderData: { items: mockNodes, total: mockNodes.length },
    retry: false,
  });

  const filteredItems = data?.items.filter((node) =>
    searchTerm
      ? node.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        node.text?.toLowerCase().includes(searchTerm.toLowerCase())
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
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Nodes</h1>
        <p className="text-muted-foreground mt-1">
          Browse and filter document nodes
        </p>
      </div>

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
                placeholder="Search by title or text..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 border-2"
              />
            </div>
            <Select value={topicFilter} onValueChange={setTopicFilter}>
              <SelectTrigger className="w-48 border-2">
                <SelectValue placeholder="All Topics" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Topics</SelectItem>
                <SelectItem value="arxiv">ArXiv</SelectItem>
                <SelectItem value="pubmed">PubMed</SelectItem>
                <SelectItem value="research">Research</SelectItem>
                <SelectItem value="technical">Technical</SelectItem>
                <SelectItem value="academic">Academic</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card className="border-2">
        <Table>
          <TableHeader>
            <TableRow className="border-b-2">
              <TableHead className="font-mono uppercase">ID</TableHead>
              <TableHead className="font-mono uppercase">Title</TableHead>
              <TableHead className="font-mono uppercase">Topic</TableHead>
              <TableHead className="font-mono uppercase">Category</TableHead>
              <TableHead className="font-mono uppercase">Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredItems?.map((node) => (
              <TableRow key={node.id} className="border-b-2">
                <TableCell className="font-mono text-xs">{node.id}</TableCell>
                <TableCell className="max-w-md truncate font-medium">
                  {node.title || "Untitled"}
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className="border-2">
                    {node.topic}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge variant="secondary" className="border-2">
                    {node.category}
                  </Badge>
                </TableCell>
                <TableCell className="font-mono text-xs">
                  {new Date(node.created_at).toLocaleDateString()}
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

export default Nodes;
