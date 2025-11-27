# DevForge Studio

A modern frontend dashboard for the DevForge hybrid vector + graph retrieval backend. Built with React, TypeScript, and Tailwind CSS.

## Features

- **Overview Dashboard**: Real-time system stats, health monitoring, and snapshot information
- **Nodes Explorer**: Paginated browsing with filters for topics and categories
- **Edges Explorer**: Relationship visualization with type-based filtering
- **Hybrid Search**: Advanced search combining vector similarity and graph relationships
- **Ingestion Tools**: Commands and guides for data import
- **Analytics**: Charts and data distribution visualization

## Tech Stack

- **React 18** + **TypeScript** - Modern, type-safe UI
- **Vite** - Fast build tooling
- **Tailwind CSS** - Utility-first styling with brutalist design system
- **React Query** (TanStack Query) - Data fetching and caching
- **Recharts** - Data visualization
- **Axios** - HTTP client
- **shadcn/ui** - Accessible component library

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- DevForge backend running (or use mock data)

### Installation

```sh
# Install dependencies
npm install

# Copy environment template
cp .env.example .env

# Update .env with your backend URL
# VITE_API_URL=http://localhost:8000
```

### Development

```sh
# Start development server
npm run dev

# Open http://localhost:8080
```

The dashboard will automatically use mock data if the backend is not available, allowing you to explore the UI without a running backend.

### Environment Variables

- `VITE_API_URL` - Backend API URL (default: `http://localhost:8000`)

## Project Structure

```
src/
├── components/        # Reusable UI components
│   ├── ui/           # shadcn/ui components
│   ├── AppSidebar.tsx
│   └── StatCard.tsx
├── pages/            # Page components
│   ├── Overview.tsx
│   ├── Nodes.tsx
│   ├── Edges.tsx
│   ├── Search.tsx
│   ├── Ingestion.tsx
│   └── Stats.tsx
├── lib/              # Utilities and API clients
│   ├── api.ts        # API service layer
│   ├── mockData.ts   # Mock data for demo
│   └── utils.ts
├── types/            # TypeScript type definitions
│   └── index.ts
└── App.tsx           # Root component with routing
```

## API Integration

The dashboard connects to the DevForge backend API with the following endpoints:

### Nodes
- `GET /nodes?limit&offset&topic&category` - List nodes with pagination
- `GET /nodes/{id}` - Get node details
- `GET /nodes/{id}/neighbors?limit=50` - Get connected nodes

### Edges
- `GET /edges?type&limit&offset` - List edges with filters

### Search
- `POST /search/hybrid` - Hybrid vector + graph search
  ```json
  {
    "query": "machine learning",
    "vector_weight": 0.7,
    "graph_weight": 0.3
  }
  ```

### Stats
- `GET /stats` - System statistics
- `GET /healthz` - Health check

## Backend Setup

To run the full system with real data, you'll need the DevForge backend:

```sh
# Ingest sample data
python -m scripts.ingest_bulk -l 10000 -t arxiv
python -m scripts.ingest_bulk -l 10000 -t pubmed

# Generate and ingest topics
python -m scripts.generate_topics_data --per-topic 100 --output data/general_topics.json
python -m scripts.ingest_bulk --source json --file data/general_topics.json -l 5000
```

## Build for Production

```sh
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

## Design System

The dashboard uses a brutalist design theme with:
- Space Grotesk font family
- High contrast black/white color scheme
- Bold borders and shadows
- Monospace fonts for data display
- Semantic color tokens for consistency

All styling follows the design system defined in `src/index.css` and `tailwind.config.ts`.

## License

MIT
