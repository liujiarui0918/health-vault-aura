import { useState } from "react";
import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, Filter, MessageSquare, User, ChevronDown } from "lucide-react";

const mockProfiles = [
  {
    id: "anon_9341",
    conditions: ["Type 2 Diabetes", "Hypertension"],
    meds: ["Metformin", "Lisinopril"],
    ageRange: "40–45",
    sex: "Male",
    matchScore: 94,
    wearable: true,
  },
  {
    id: "anon_1827",
    conditions: ["Type 2 Diabetes", "GERD"],
    meds: ["Metformin", "Omeprazole"],
    ageRange: "35–40",
    sex: "Female",
    matchScore: 87,
    wearable: false,
  },
  {
    id: "anon_6482",
    conditions: ["Hypertension", "High Cholesterol"],
    meds: ["Lisinopril", "Atorvastatin"],
    ageRange: "45–50",
    sex: "Male",
    matchScore: 82,
    wearable: true,
  },
  {
    id: "anon_3014",
    conditions: ["Type 2 Diabetes"],
    meds: ["Metformin", "Januvia"],
    ageRange: "50–55",
    sex: "Male",
    matchScore: 79,
    wearable: false,
  },
];

const filters = [
  { label: "Condition", options: ["Type 2 Diabetes", "Hypertension", "High Cholesterol", "GERD"] },
  { label: "Age Range", options: ["18–25", "25–35", "35–40", "40–45", "45–50", "50+"] },
  { label: "Sex", options: ["Male", "Female", "Other"] },
];

export default function SearchProfiles() {
  const [search, setSearch] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);

  return (
    <AppLayout title="Search Profiles">
      <div className="p-6 space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">Search Health Profiles</h2>
          <p className="text-muted-foreground text-sm">Find similar profiles by condition, medication, or demographics. All profiles are pseudonymous and de-identified.</p>
        </div>

        {/* Search bar */}
        <div className="flex gap-3 flex-wrap">
          <div className="relative flex-1 min-w-[240px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search by condition, medication, or keyword..."
              className="pl-10 h-11"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <Button
            variant="outline"
            className="gap-2"
            onClick={() => setFilterOpen(!filterOpen)}
          >
            <Filter className="w-4 h-4" />
            Filters
            <ChevronDown className={`w-4 h-4 transition-transform ${filterOpen ? "rotate-180" : ""}`} />
          </Button>
          <Button className="gap-2 bg-primary text-primary-foreground hover:opacity-90 shadow-teal">
            <Search className="w-4 h-4" />
            Search
          </Button>
        </div>

        {/* Filters panel */}
        {filterOpen && (
          <div className="vault-card p-4 flex flex-wrap gap-6">
            {filters.map((f) => (
              <div key={f.label}>
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider block mb-2">{f.label}</label>
                <div className="flex flex-wrap gap-1.5">
                  {f.options.map((opt) => (
                    <button
                      key={opt}
                      className="px-2.5 py-1 rounded-lg text-xs font-medium bg-muted hover:bg-accent text-foreground transition-colors"
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Results */}
        <div className="text-sm text-muted-foreground">{mockProfiles.length} profiles found matching your criteria</div>

        <div className="grid md:grid-cols-2 gap-4">
          {mockProfiles.map((profile) => (
            <div key={profile.id} className="vault-card-hover p-5">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-accent flex items-center justify-center">
                    <User className="w-5 h-5 text-accent-foreground" />
                  </div>
                  <div>
                    <div className="font-mono font-semibold text-foreground">{profile.id}</div>
                    <div className="text-xs text-muted-foreground">{profile.ageRange} · {profile.sex}</div>
                  </div>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="text-xs font-medium text-primary">{profile.matchScore}%</div>
                  <div className="text-[10px] text-muted-foreground">match</div>
                </div>
              </div>

              <div className="space-y-2 mb-4">
                <div>
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Conditions</div>
                  <div className="flex flex-wrap gap-1">
                    {profile.conditions.map((c) => (
                      <span key={c} className="px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200 text-xs">{c}</span>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Medications</div>
                  <div className="flex flex-wrap gap-1">
                    {profile.meds.map((m) => (
                      <span key={m} className="px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs">{m}</span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button size="sm" className="flex-1 gap-1.5 bg-primary text-primary-foreground hover:opacity-90">
                  <MessageSquare className="w-3 h-3" /> Request Message
                </Button>
                {profile.wearable && (
                  <span className="status-active text-[10px]">Wearable Data</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </AppLayout>
  );
}
