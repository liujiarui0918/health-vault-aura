import { Link } from "react-router-dom";
import { Activity, Users, Shield, FileText, AlertTriangle, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const stats = [
  { label: "Total Users", value: "14,281", icon: Users },
  { label: "Institutions", value: "47", icon: Shield },
  { label: "Active Campaigns", value: "23", icon: FileText },
  { label: "Pending Reviews", value: "8", icon: AlertTriangle },
];

export default function AdminDashboard() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
            <Activity className="w-4 h-4 text-primary" />
          </div>
          <span className="font-bold">HealthVault Admin</span>
        </div>
        <Link to="/admin/audit"><Button variant="outline" size="sm">Audit Logs</Button></Link>
      </header>
      <div className="p-6 space-y-6 max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold text-foreground">Admin Dashboard</h1>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((s) => (
            <div key={s.label} className="vault-card p-5">
              <s.icon className="w-5 h-5 text-muted-foreground mb-2" />
              <div className="text-2xl font-bold text-foreground">{s.value}</div>
              <div className="text-xs text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
        <div className="vault-card p-6">
          <h2 className="font-semibold text-foreground mb-4">Pending Institution Reviews</h2>
          <div className="space-y-2">
            {["Johns Hopkins Research", "UCLA Medical Center"].map((inst) => (
              <div key={inst} className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                <span className="text-sm text-foreground">{inst}</span>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">Reject</Button>
                  <Button size="sm" className="bg-primary text-primary-foreground">Approve</Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
