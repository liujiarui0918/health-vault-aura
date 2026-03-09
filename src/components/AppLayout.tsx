import { ReactNode, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard, Upload, Clock, Search, MessageSquare,
  Mail, Shield, Settings, LogOut, ChevronLeft, ChevronRight,
  Activity, Wallet, Bell, Menu, X
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "Dashboard", icon: LayoutDashboard, href: "/dashboard" },
  { label: "Health Data", icon: Upload, href: "/health-data" },
  { label: "Timeline", icon: Clock, href: "/timeline" },
  { label: "Search Profiles", icon: Search, href: "/search" },
  { label: "Messages", icon: MessageSquare, href: "/messages", badge: 3 },
  { label: "Invitations", icon: Mail, href: "/invitations", badge: 1 },
  { label: "Consent Center", icon: Shield, href: "/consent" },
];

interface AppLayoutProps {
  children: ReactNode;
  title?: string;
}

export default function AppLayout({ children, title }: AppLayoutProps) {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className={cn("flex items-center gap-3 px-4 py-5 border-b border-sidebar-border", collapsed && "justify-center")}>
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/20 border border-primary/30 flex-shrink-0">
          <Activity className="w-4 h-4 text-primary" />
        </div>
        {!collapsed && (
          <div>
            <span className="font-bold text-sidebar-accent-foreground tracking-tight">HealthVault</span>
            <div className="text-[10px] text-sidebar-foreground/60 font-mono">v1.0 · encrypted</div>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const active = location.pathname === item.href;
          return (
            <Link
              key={item.href}
              to={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150",
                collapsed && "justify-center px-2",
                active
                  ? "bg-sidebar-primary/15 text-sidebar-primary border border-sidebar-primary/20"
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              )}
              onClick={() => setMobileOpen(false)}
            >
              <item.icon className={cn("flex-shrink-0", active ? "w-4 h-4 text-primary" : "w-4 h-4")} />
              {!collapsed && (
                <span className="flex-1">{item.label}</span>
              )}
              {!collapsed && item.badge && (
                <span className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold bg-primary text-primary-foreground rounded-full">
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Wallet indicator */}
      {!collapsed && (
        <div className="mx-3 mb-3 p-3 rounded-lg bg-sidebar-accent/50 border border-sidebar-border">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-sidebar-foreground/80 font-mono truncate">0x3f…a9b2</span>
          </div>
          <div className="mt-1 text-[10px] text-sidebar-foreground/50">Wallet Connected</div>
        </div>
      )}

      {/* Bottom actions */}
      <div className={cn("px-3 py-3 border-t border-sidebar-border space-y-0.5", collapsed && "px-2")}>
        <Link to="/settings" className={cn("flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-sidebar-foreground hover:bg-sidebar-accent transition-all", collapsed && "justify-center px-2")}>
          <Settings className="w-4 h-4" />
          {!collapsed && <span>Settings</span>}
        </Link>
        <Link to="/login" className={cn("flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-sidebar-foreground hover:bg-sidebar-accent transition-all", collapsed && "justify-center px-2")}>
          <LogOut className="w-4 h-4" />
          {!collapsed && <span>Sign Out</span>}
        </Link>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Desktop Sidebar */}
      <aside
        className={cn(
          "hidden md:flex flex-col bg-sidebar-background transition-all duration-300 flex-shrink-0 relative",
          collapsed ? "w-[60px]" : "w-[220px]"
        )}
        style={{ background: "var(--gradient-navy)" }}
      >
        <SidebarContent />
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3 top-8 w-6 h-6 rounded-full bg-sidebar-background border border-sidebar-border flex items-center justify-center text-sidebar-foreground hover:bg-sidebar-accent transition-all z-10"
        >
          {collapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />}
        </button>
      </aside>

      {/* Mobile Sidebar */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setMobileOpen(false)} />
          <aside className="absolute left-0 top-0 h-full w-[220px] flex flex-col" style={{ background: "var(--gradient-navy)" }}>
            <SidebarContent />
          </aside>
        </div>
      )}

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center justify-between px-6 py-4 bg-card border-b border-border flex-shrink-0">
          <div className="flex items-center gap-3">
            <button className="md:hidden" onClick={() => setMobileOpen(true)}>
              <Menu className="w-5 h-5" />
            </button>
            {title && <h1 className="text-lg font-semibold text-foreground">{title}</h1>}
          </div>
          <div className="flex items-center gap-3">
            <button className="relative p-2 rounded-lg hover:bg-muted transition-all">
              <Bell className="w-4 h-4 text-muted-foreground" />
              <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-primary rounded-full" />
            </button>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-accent rounded-lg">
              <div className="w-6 h-6 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center text-xs font-bold text-primary">A</div>
              <span className="text-sm font-medium text-accent-foreground hidden sm:block">anon_7842</span>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
