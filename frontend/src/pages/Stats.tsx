import { useQuery } from "@tanstack/react-query";
import { statsApi, nodesApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
} from "recharts";
import { Download } from "lucide-react";

const COLORS = ["hsl(var(--chart-1))", "hsl(var(--chart-2))", "hsl(var(--chart-3))", "hsl(var(--chart-4))", "hsl(var(--chart-5))"];

const Stats = () => {
  const { data: stats } = useQuery({
    queryKey: ["stats"],
    queryFn: statsApi.get,
    retry: 1,
  });

  const { data: nodesData } = useQuery({
    queryKey: ["nodes-stats"],
    queryFn: () => nodesApi.getAll(1, 1000),
    retry: 1,
  });

  // Aggregate data for charts
  const topicData = nodesData?.items.reduce((acc: any, node) => {
    const topic = node.topic || "unknown";
    acc[topic] = (acc[topic] || 0) + 1;
    return acc;
  }, {});

  const topicChartData = Object.entries(topicData || {}).map(([name, value]) => ({
    name,
    value,
  }));

  const categoryData = nodesData?.items.reduce((acc: any, node) => {
    const category = node.category || "unknown";
    acc[category] = (acc[category] || 0) + 1;
    return acc;
  }, {});

  const categoryChartData = Object.entries(categoryData || {}).map(([name, count]) => ({
    name,
    count,
  }));

  const handleExport = () => {
    const csv = [
      "ID,Title,Topic,Category,Created",
      ...(nodesData?.items.map((n) =>
        [n.id, n.title, n.topic, n.category, n.created_at].join(",")
      ) || []),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "nodes_export.csv";
    a.click();
  };

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Statistics</h1>
          <p className="text-muted-foreground mt-1">
            Analytics and data distribution
          </p>
        </div>
        <Button onClick={handleExport} variant="outline" className="border-2">
          <Download className="mr-2 h-4 w-4" />
          Export CSV
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="font-mono">Topic Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={topicChartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) =>
                    `${name}: ${(percent * 100).toFixed(0)}%`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {topicChartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-2">
          <CardHeader>
            <CardTitle className="font-mono">Category Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={categoryChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="hsl(var(--primary))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card className="border-2">
        <CardHeader>
          <CardTitle className="font-mono">System Metrics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 border-2 border-border">
              <div className="text-sm text-muted-foreground mb-1">
                Total Documents
              </div>
              <div className="text-2xl font-bold font-mono">
                {stats?.nodes.toLocaleString()}
              </div>
            </div>
            <div className="p-4 border-2 border-border">
              <div className="text-sm text-muted-foreground mb-1">
                Total Edges
              </div>
              <div className="text-2xl font-bold font-mono">
                {stats?.edges.toLocaleString()}
              </div>
            </div>
            <div className="p-4 border-2 border-border">
              <div className="text-sm text-muted-foreground mb-1">
                Avg Connections
              </div>
              <div className="text-2xl font-bold font-mono">
                {stats?.graph_degree_avg?.toFixed(2)}
              </div>
            </div>
            <div className="p-4 border-2 border-border">
              <div className="text-sm text-muted-foreground mb-1">
                Index Size
              </div>
              <div className="text-2xl font-bold font-mono">
                {stats?.vector_index_size
                  ? `${(stats.vector_index_size / 1024 / 1024).toFixed(1)}MB`
                  : "N/A"}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Stats;
