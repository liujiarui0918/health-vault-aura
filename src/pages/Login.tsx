import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Activity, ArrowRight, Wallet } from "lucide-react";

export default function Login() {
  return (
    <div className="min-h-screen bg-background flex">
      {/* Left panel */}
      <div className="hidden lg:flex flex-col justify-between w-[480px] flex-shrink-0 p-12" style={{ background: "var(--gradient-hero)" }}>
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/20 border border-primary/30">
            <Activity className="w-4 h-4 text-primary" />
          </div>
          <span className="font-bold text-white">HealthVault</span>
        </div>
        <div>
          <blockquote className="text-white/80 text-lg leading-relaxed italic mb-4">
            "Finally, a platform where I control exactly who sees my health data. The consent system is revolutionary."
          </blockquote>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-teal-600/30 border border-teal-600/40 flex items-center justify-center text-sm font-bold text-teal-400">A</div>
            <div>
              <div className="text-sm font-medium text-white">anon_4821</div>
              <div className="text-xs text-white/50">Individual User · 2 years on platform</div>
            </div>
          </div>
        </div>
        <div className="text-xs text-white/30 font-mono">End-to-End Encrypted · Privacy First</div>
      </div>

      {/* Right panel */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-sm">
          <div className="mb-8">
            <Link to="/" className="flex items-center gap-2 mb-8 lg:hidden">
              <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-primary/10 border border-primary/20">
                <Activity className="w-3.5 h-3.5 text-primary" />
              </div>
              <span className="font-bold text-sm">HealthVault</span>
            </Link>
            <h1 className="text-2xl font-bold text-foreground mb-1">Welcome back</h1>
            <p className="text-muted-foreground text-sm">Sign in to your encrypted vault</p>
          </div>

          <div className="space-y-4">
            <div>
              <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5 block">Email</Label>
              <Input type="email" placeholder="you@example.com" className="h-11" />
            </div>
            <div>
              <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5 block">Password</Label>
              <Input type="password" placeholder="••••••••" className="h-11" />
            </div>

            <Link to="/dashboard">
              <Button className="w-full h-11 bg-primary text-primary-foreground hover:opacity-90 shadow-teal gap-2 mt-2">
                Sign In <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border" />
              </div>
              <div className="relative flex justify-center text-xs text-muted-foreground bg-background px-2">or</div>
            </div>

            <Button variant="outline" className="w-full h-11 gap-2">
              <Wallet className="w-4 h-4" />
              Connect Wallet to Sign In
            </Button>
          </div>

          <p className="text-center text-sm text-muted-foreground mt-6">
            Don't have an account?{" "}
            <Link to="/signup" className="text-primary font-medium hover:underline">Create one</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
