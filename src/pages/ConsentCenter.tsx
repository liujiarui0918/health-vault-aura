import { useState } from "react";
import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Shield, Lock, Eye, EyeOff, Building2, CheckCircle2, AlertTriangle, ChevronRight } from "lucide-react";

const dataCategories = [
  { label: "Conditions & Diagnoses", searchable: true, count: 2 },
  { label: "Medications", searchable: true, count: 3 },
  { label: "Lab Results", searchable: false, count: 8 },
  { label: "Vitals", searchable: true, count: 4 },
  { label: "Wearable Data", searchable: true, count: 1 },
  { label: "Imaging / Reports", searchable: false, count: 6 },
];

const activeConsents = [
  {
    institution: "Stanford University School of Medicine",
    study: "Long-term Outcomes of T2D Management with GLP-1 Agonists",
    granted: "Mar 6, 2025",
    expires: "Sep 6, 2025",
    scope: ["Conditions", "Medications", "Lab Results"],
  },
  {
    institution: "Mayo Clinic Research",
    study: "Hypertension and Cardiovascular Risk in T2D Patients",
    granted: "Feb 15, 2025",
    expires: "Aug 15, 2025",
    scope: ["Conditions", "Medications", "Vitals", "Wearable Data"],
  },
];

export default function ConsentCenter() {
  const [profileSearchable, setProfileSearchable] = useState(true);
  const [categories, setCategories] = useState(dataCategories);

  const toggleCategory = (label: string) => {
    setCategories(categories.map(c => c.label === label ? { ...c, searchable: !c.searchable } : c));
  };

  return (
    <AppLayout title="Consent Center">
      <div className="p-6 space-y-6 max-w-4xl">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">Consent Center</h2>
          <p className="text-muted-foreground text-sm">Manage what's searchable, view active consents, and control your data sharing.</p>
        </div>

        {/* Profile visibility */}
        <div className="vault-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-accent flex items-center justify-center">
                {profileSearchable ? <Eye className="w-5 h-5 text-accent-foreground" /> : <EyeOff className="w-5 h-5 text-muted-foreground" />}
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Profile Searchability</h3>
                <p className="text-sm text-muted-foreground">Allow others to find you based on your de-identified health profile</p>
              </div>
            </div>
            <Switch checked={profileSearchable} onCheckedChange={setProfileSearchable} />
          </div>

          {profileSearchable && (
            <div className="pt-4 border-t border-border">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Data Categories (Toggle searchability)</div>
              <div className="grid md:grid-cols-2 gap-2">
                {categories.map((cat) => (
                  <div key={cat.label} className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors">
                    <div className="flex items-center gap-2">
                      {cat.searchable ? <Eye className="w-3.5 h-3.5 text-primary" /> : <EyeOff className="w-3.5 h-3.5 text-muted-foreground" />}
                      <span className="text-sm text-foreground">{cat.label}</span>
                      <span className="text-xs text-muted-foreground">({cat.count})</span>
                    </div>
                    <Switch checked={cat.searchable} onCheckedChange={() => toggleCategory(cat.label)} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Active consents */}
        <div className="vault-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-foreground flex items-center gap-2">
              <Shield className="w-4 h-4 text-primary" /> Active Consent Grants
            </h3>
            <span className="text-xs text-muted-foreground">{activeConsents.length} active</span>
          </div>

          <div className="space-y-3">
            {activeConsents.map((consent, i) => (
              <div key={i} className="flex items-start gap-4 p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-colors">
                <div className="w-10 h-10 rounded-xl bg-background border border-border flex items-center justify-center flex-shrink-0">
                  <Building2 className="w-5 h-5 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-foreground text-sm truncate">{consent.institution}</span>
                    <span className="status-active"><CheckCircle2 className="w-3 h-3" /> Active</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5 truncate">{consent.study}</div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    <div className="text-[10px] text-muted-foreground">Granted: {consent.granted}</div>
                    <div className="text-[10px] text-muted-foreground">Expires: {consent.expires}</div>
                  </div>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {consent.scope.map((s) => (
                      <span key={s} className="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-medium">{s}</span>
                    ))}
                  </div>
                </div>
                <Button variant="ghost" size="sm" className="flex-shrink-0 text-destructive hover:bg-destructive/10 text-xs">
                  Revoke
                </Button>
              </div>
            ))}
          </div>
        </div>

        {/* De-ID notice */}
        <div className="vault-card p-6 border-primary/20 bg-accent/30">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-foreground mb-1">What gets shared?</h3>
              <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                Only de-identified data is ever shared. Names, exact dates of birth, addresses, and other PII are <strong>automatically stripped</strong> before any data is made visible. Your age is shown as a 5-year range (e.g., 35–40), not your exact age.
              </p>
              <Button variant="link" className="p-0 h-auto text-primary text-sm gap-1">
                View De-ID Preview <ChevronRight className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </div>

        {/* Audit anchor */}
        <div className="flex items-center gap-3 p-4 rounded-xl bg-muted/50 border border-border">
          <Lock className="w-4 h-4 text-primary" />
          <div className="text-sm text-muted-foreground">
            <strong className="text-foreground">All consents are anchored on-chain.</strong> This provides immutable proof of when consent was granted, modified, or revoked.
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
