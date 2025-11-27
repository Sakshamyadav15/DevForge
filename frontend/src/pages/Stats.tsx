import { useQuery } from "@tanstack/react-query";
import { statsApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import {
  Download,
  Database,
  GitBranch,
  Layers,
  Tag,
  FolderOpen,
  Cpu,
  RefreshCw,
} from "lucide-react";

const COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
  "#8884d8",
  "#82ca9d",
  "#ffc658",
];

const Stats = () => {
  const {
    data: stats,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ["stats"],
    queryFn: statsApi.get,
    retry: 1,
    staleTime: 30000,
  });

  const handleExportStats = () => {
    if (!stats) return;

    const exportData = {
      exported_at: new Date().toISOString(),
      summary: {
        total_nodes: stats.nodes,
        total_edges: stats.edges,
        unique_topics: stats.unique_topics,
        unique_categories: stats.unique_categories,
        unique_sources: stats.unique_sources,
        avg_degree: stats.avg_degree,
        avg_edge_weight: stats.avg_edge_weight,
      },
      topic_distribution: stats.topic_distribution,
      category_distribution: stats.category_distribution,
      source_distribution: stats.source_distribution,
      edge_type_distribution: stats.edge_type_distribution,
      embedding_info: stats.embedding,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `devforge_stats_${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="p-8 space-y-6">
        <Skeleton className="h-12 w-64" />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
        <div className="grid gap-6 md:grid-cols-2">
          <Skeleton className="h-80" />
          <Skeleton className="h-80" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 space-y-6">
        <h1 className="text-3xl font-bold tracking-tight">Statistics</h1>
        <Alert className="border-2 border-red-500">
          <AlertDescription>
            Failed to load statistics. Make sure the backend is running.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Statistics</h1>
          <p className="text-muted-foreground mt-1">
            Real-time analytics and data distribution from your knowledge graph
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => refetch()}
            variant="outline"
            className="border-2"
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <Button
            onClick={handleExportStats}
            variant="outline"
            className="border-2"
          >
            <Download className="mr-2 h-4 w-4" />
            Export JSON
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  Total Nodes
                </p>
                <p className="text-3xl font-bold font-mono">
                  {stats?.nodes.toLocaleString()}
                </p>
              </div>
              <Database className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  Total Edges
                </p>
                <p className="text-3xl font-bold font-mono">
                  {stats?.edges.toLocaleString()}
                </p>
              </div>
              <GitBranch className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  Avg Degree
                </p>
                <p className="text-3xl font-bold font-mono">
                  {stats?.avg_degree?.toFixed(2) || "0.00"}
                </p>
                <p className="text-xs text-muted-foreground">edges per node</p>
              </div>
              <Layers className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  Vector Index
                </p>
                <p className="text-3xl font-bold font-mono">
                  {stats?.vector_index_size?.toLocaleString() || 0}
                </p>
                <p className="text-xs text-muted-foreground">vectors indexed</p>
              </div>
              <Cpu className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Secondary Metrics */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Tag className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  Unique Topics
                </p>
                <p className="text-2xl font-bold font-mono">
                  {stats?.unique_topics || 0}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <FolderOpen className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  Unique Categories
                </p>
                <p className="text-2xl font-bold font-mono">
                  {stats?.unique_categories || 0}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Database className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  Unique Sources
                </p>
                <p className="text-2xl font-bold font-mono">
                  {stats?.unique_sources || 0}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Topic Distribution Pie Chart */}
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="font-mono text-sm uppercase tracking-wider">
              Topic Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {stats?.topic_distribution && stats.topic_distribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={stats.topic_distribution.slice(0, 8)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      percent > 0.05 ? `${(percent * 100).toFixed(0)}%` : ""
                    }
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {stats.topic_distribution.slice(0, 8).map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[index % COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number) => [
                      value.toLocaleString(),
                      "Documents",
                    ]}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                No topic data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Category Distribution Bar Chart */}
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="font-mono text-sm uppercase tracking-wider">
              Category Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            {stats?.category_distribution &&
            stats.category_distribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={stats.category_distribution.slice(0, 10)}
                  layout="vertical"
                  margin={{ left: 80 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis
                    dataKey="name"
                    type="category"
                    width={75}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip
                    formatter={(value: number) => [
                      value.toLocaleString(),
                      "Documents",
                    ]}
                  />
                  <Bar dataKey="count" fill="hsl(var(--primary))" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                No category data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Source Distribution */}
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="font-mono text-sm uppercase tracking-wider">
              Data Sources
            </CardTitle>
          </CardHeader>
          <CardContent>
            {stats?.source_distribution &&
            stats.source_distribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.source_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value: number) => [
                      value.toLocaleString(),
                      "Documents",
                    ]}
                  />
                  <Bar dataKey="count" fill="hsl(var(--chart-2))" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[250px] flex items-center justify-center text-muted-foreground">
                No source data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Edge Type Distribution */}
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="font-mono text-sm uppercase tracking-wider">
              Edge Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            {stats?.edge_type_distribution &&
            stats.edge_type_distribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.edge_type_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value: number) => [
                      value.toLocaleString(),
                      "Edges",
                    ]}
                  />
                  <Bar dataKey="count" fill="hsl(var(--chart-3))" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[250px] flex items-center justify-center text-muted-foreground">
                No edge type data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* System Information */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider">
            System Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="p-4 border-2 border-border rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">
                Embedding Model
              </p>
              <p className="font-mono text-sm font-medium truncate">
                {stats?.embedding?.model || "N/A"}
              </p>
            </div>
            <div className="p-4 border-2 border-border rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">
                Embedding Dimension
              </p>
              <p className="font-mono text-sm font-medium">
                {stats?.embedding?.dimension || "N/A"}
              </p>
            </div>
            <div className="p-4 border-2 border-border rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">
                Avg Edge Weight
              </p>
              <p className="font-mono text-sm font-medium">
                {stats?.avg_edge_weight?.toFixed(4) || "N/A"}
              </p>
            </div>
            <div className="p-4 border-2 border-border rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">Last Updated</p>
              <p className="font-mono text-sm font-medium">
                {stats?.snapshot_last_updated
                  ? new Date(stats.snapshot_last_updated).toLocaleString()
                  : "N/A"}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Stats;
