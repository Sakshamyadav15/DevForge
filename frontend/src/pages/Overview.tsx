import { useQuery } from "@tanstack/react-query";
import { statsApi } from "@/lib/api";
import { StatCard } from "@/components/StatCard";
import { Database, GitBranch, Clock, HardDrive, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription } from "@/components/ui/alert";

const Overview = () => {
  const { toast } = useToast();

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

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: statsApi.health,
    refetchInterval: 30000,
    retry: false,
  });

  const handleRefresh = async () => {
    await refetch();
    toast({
      title: "Refreshed",
      description: "Stats updated successfully",
    });
  };

  if (isLoading) {
    return (
      <div className="p-8 space-y-6">
        <Skeleton className="h-12 w-64" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Overview</h1>
          <p className="text-muted-foreground mt-1">
            DevForge hybrid vector + graph retrieval system
          </p>
        </div>
        <Button onClick={handleRefresh} variant="outline" className="border-2">
          Refresh
        </Button>
      </div>

      {error && (
        <Alert className="border-2">
          <AlertDescription>
            Unable to connect to backend. Showing mock data for demonstration.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <StatCard
          title="Total Nodes"
          value={stats?.nodes.toLocaleString() || 0}
          icon={Database}
          description="Documents indexed"
        />
        <StatCard
          title="Total Edges"
          value={stats?.edges.toLocaleString() || 0}
          icon={GitBranch}
          description="Relationships mapped"
        />
        <StatCard
          title="Index Size"
          value={
            stats?.vector_index_size
              ? `${stats.vector_index_size.toLocaleString()}`
              : "N/A"
          }
          icon={HardDrive}
          description="Vectors indexed"
        />
        <StatCard
          title="Avg Degree"
          value={stats?.avg_degree?.toFixed(2) || "N/A"}
          icon={GitBranch}
          description="Connections per node"
        />
        <StatCard
          title="Status"
          value={health?.status === "ok" ? "Healthy" : "Error"}
          icon={Activity}
          description={`Updated ${new Date(stats?.snapshot_last_updated || "").toLocaleDateString()}`}
        />
      </div>

      <Card className="border-2">
        <CardHeader>
          <CardTitle className="font-mono">Quick Stats</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex justify-between items-center p-4 border-2 border-border">
            <span className="font-medium">Last Snapshot Update</span>
            <span className="font-mono text-sm">
              {new Date(stats?.snapshot_last_updated || "").toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between items-center p-4 border-2 border-border">
            <span className="font-medium">Edge Density</span>
            <span className="font-mono text-sm">
              {stats ? (stats.edges / stats.nodes).toFixed(2) : "N/A"} edges/node
            </span>
          </div>
          <div className="flex justify-between items-center p-4 border-2 border-border">
            <span className="font-medium">System Status</span>
            <span className={`font-bold ${health?.status === "ok" ? "text-green-600" : "text-red-600"}`}>
              {health?.status?.toUpperCase() || "UNKNOWN"}
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Overview;
