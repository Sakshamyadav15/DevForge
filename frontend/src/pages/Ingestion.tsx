import { useQuery } from "@tanstack/react-query";
import { statsApi } from "@/lib/api";
import { mockStats } from "@/lib/mockData";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Code, Terminal } from "lucide-react";

const Ingestion = () => {
  const { data: stats } = useQuery({
    queryKey: ["stats"],
    queryFn: statsApi.get,
    placeholderData: mockStats,
    retry: false,
  });

  const commands = [
    {
      title: "Ingest ArXiv Papers",
      command: "python -m scripts.ingest_bulk -l 10000 -t arxiv",
      description: "Import 10,000 papers from ArXiv",
    },
    {
      title: "Ingest PubMed Papers",
      command: "python -m scripts.ingest_bulk -l 10000 -t pubmed",
      description: "Import 10,000 papers from PubMed",
    },
    {
      title: "Generate Topics Data",
      command:
        "python -m scripts.generate_topics_data --per-topic 100 --output data/general_topics.json",
      description: "Generate topic-based synthetic data",
    },
    {
      title: "Ingest JSON Data",
      command:
        "python -m scripts.ingest_bulk --source json --file data/general_topics.json -l 5000",
      description: "Import data from JSON file",
    },
  ];

  return (
    <div className="p-8 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Ingestion</h1>
        <p className="text-muted-foreground mt-1">
          Run ingestion scripts to populate the database
        </p>
      </div>

      <Alert className="border-2">
        <AlertDescription>
          Run these commands in your terminal from the backend project directory.
          The dashboard will automatically reflect new data once ingestion
          completes.
        </AlertDescription>
      </Alert>

      <Card className="border-2">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider">
            Current Snapshot
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex justify-between items-center p-4 border-2 border-border">
            <span className="font-medium">Total Nodes</span>
            <span className="font-mono text-xl font-bold">
              {stats?.nodes.toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between items-center p-4 border-2 border-border">
            <span className="font-medium">Total Edges</span>
            <span className="font-mono text-xl font-bold">
              {stats?.edges.toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between items-center p-4 border-2 border-border">
            <span className="font-medium">Last Updated</span>
            <span className="font-mono text-sm">
              {new Date(stats?.snapshot_last_updated || "").toLocaleString()}
            </span>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-4">
        <h2 className="text-xl font-bold font-mono">Ingestion Commands</h2>
        {commands.map((cmd, index) => (
          <Card key={index} className="border-2">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Terminal className="h-4 w-4" />
                {cmd.title}
              </CardTitle>
              <p className="text-sm text-muted-foreground">{cmd.description}</p>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-muted p-3 border-2 border-border font-mono text-sm overflow-x-auto">
                  {cmd.command}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    navigator.clipboard.writeText(cmd.command);
                  }}
                  className="border-2"
                >
                  <Code className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Alert className="border-2">
        <AlertDescription className="text-xs font-mono">
          Tip: These scripts should be run from your backend environment where
          Python and required dependencies are installed.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default Ingestion;
