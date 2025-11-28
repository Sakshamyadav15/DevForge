import { Database, GitBranch, Search, Activity, Upload, BarChart3 } from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";

const items = [
  { title: "Overview", url: "/", icon: Activity },
  { title: "Nodes", url: "/nodes", icon: Database },
  { title: "Edges", url: "/edges", icon: GitBranch },
  { title: "Search", url: "/search", icon: Search },
  { title: "Ingestion", url: "/ingestion", icon: Upload },
  { title: "Stats", url: "/stats", icon: BarChart3 },
];

export function AppSidebar() {
  const { open } = useSidebar();
  const location = useLocation();
  const currentPath = location.pathname;

  const isActive = (path: string) => {
    if (path === "/") return currentPath === "/";
    return currentPath.startsWith(path);
  };

  return (
    <Sidebar collapsible="icon" className="border-r-2 border-border">
      <SidebarContent>
        <div className="p-4 border-b-2 border-border">
          <h1 className="text-lg font-bold tracking-tight">
            {open ? "HybridDB" : "HD"}
          </h1>
        </div>
        <SidebarGroup>
          <SidebarGroupLabel className="text-xs font-mono uppercase tracking-wider">
            Navigation
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={isActive(item.url)}>
                    <NavLink
                      to={item.url}
                      end={item.url === "/"}
                      className="font-medium"
                    >
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
