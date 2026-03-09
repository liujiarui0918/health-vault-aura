import { Link } from "react-router-dom";
import { Activity, ArrowLeft, Shield, Lock, FileText, MessageSquare } from "lucide-react";

const logs = [
  { time: "Mar 9, 10:42", action: "Consent granted", user: "anon_7842", details: "Stanford Research — T2D Study", icon: Shield },
  { time: "Mar 9, 09:15", action: "Profile updated", user: "anon_9341", details: "Searchability toggled", icon: Lock },
  { time: "Mar 8, 18:22", action: "Data uploaded", user: "anon_1827", details: "Blood panel parsed", icon: FileText },
  { time: "Mar 8, 14:08", action: "Message sent", user: "anon_6482", details: "To anon_3014", icon: MessageSquare },
  { time: "Mar 8, 11:30", action: "Consent revoked", user: "anon_3014", details: "Mayo Clinic Research", icon: Shield },
];

export default function AuditLogs() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card px-6 py-4 flex items-center gap-4">
        <Link to="/admin" className="p-2 rounded-lg hover:bg-muted"><ArrowLeft className="w-4 h-4" /></Link>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
            <Activity className="w-4 h-4 text-primary" />
          </div>
          <span className="font-bold">Audit Logs</span>
        </div>
      </header>
      <div className="p-6 max-w-4xl mx-auto">
        <p className="text-muted-foreground text-sm mb-6">All critical actions are anchored on-chain for transparency.</p>
        <div className="vault-card divide-y divide-border">
          {logs.map((log, i) => (
            <div key={i} className="flex items-center gap-4 p-4">
              <div className="w-9 h-9 rounded-xl bg-accent flex items-center justify-center flex-shrink-0">
                <log.icon className="w-4 h-4 text-accent-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-foreground">{log.action}</div>
                <div className="text-xs text-muted-foreground">{log.user} · {log.details}</div>
              </div>
              <div className="text-xs text-muted-foreground flex-shrink-0">{log.time}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
