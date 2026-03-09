import { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const navLinks = [
  { label: "Features", href: "/#features" },
  { label: "Privacy", href: "/privacy" },
  { label: "Pricing", href: "/pricing" },
  { label: "For Institutions", href: "/institutions" },
  { label: "FAQ", href: "/faq" },
];

interface PublicLayoutProps {
  children: ReactNode;
}

export default function PublicLayout({ children }: PublicLayoutProps) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Navbar */}
      <header className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-card/80 backdrop-blur-md">
        <div className="container mx-auto flex items-center justify-between h-16 px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10 border border-primary/20">
              <Activity className="w-4 h-4 text-primary" />
            </div>
            <span className="font-bold text-foreground tracking-tight">HealthVault</span>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                to={link.href}
                className={cn(
                  "px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
                  location.pathname === link.href
                    ? "text-primary bg-accent"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                )}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <Link to="/login">
              <Button variant="ghost" size="sm">Sign In</Button>
            </Link>
            <Link to="/signup">
              <Button size="sm" className="bg-primary text-primary-foreground hover:opacity-90 shadow-teal">
                Get Started
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1 pt-16">{children}</main>

      {/* Footer */}
      <footer className="border-t border-border bg-card mt-16">
        <div className="container mx-auto px-6 py-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-primary/10 border border-primary/20">
                  <Activity className="w-3.5 h-3.5 text-primary" />
                </div>
                <span className="font-bold text-sm">HealthVault</span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Privacy-preserving personal health data, owned by you.
              </p>
            </div>
            <div>
              <div className="text-xs font-semibold text-foreground mb-3 uppercase tracking-wider">Platform</div>
              <div className="space-y-2">
                {["How it Works", "Privacy", "Security", "Roadmap"].map(l => (
                  <div key={l} className="text-sm text-muted-foreground hover:text-foreground cursor-pointer">{l}</div>
                ))}
              </div>
            </div>
            <div>
              <div className="text-xs font-semibold text-foreground mb-3 uppercase tracking-wider">For Institutions</div>
              <div className="space-y-2">
                {["Research Portal", "Pricing", "API Docs", "Compliance"].map(l => (
                  <div key={l} className="text-sm text-muted-foreground hover:text-foreground cursor-pointer">{l}</div>
                ))}
              </div>
            </div>
            <div>
              <div className="text-xs font-semibold text-foreground mb-3 uppercase tracking-wider">Legal</div>
              <div className="space-y-2">
                {["Terms", "Privacy Policy", "HIPAA Notice", "Cookie Policy"].map(l => (
                  <div key={l} className="text-sm text-muted-foreground hover:text-foreground cursor-pointer">{l}</div>
                ))}
              </div>
            </div>
          </div>
          <div className="mt-8 pt-6 border-t border-border flex flex-col md:flex-row items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">© 2025 HealthVault. All rights reserved.</div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs text-muted-foreground font-mono">System Operational</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
