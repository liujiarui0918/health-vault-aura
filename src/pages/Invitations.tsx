import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { FlaskConical, CheckCircle2, X, Calendar, Building2, FileText, ArrowRight } from "lucide-react";

const invitations = [
  {
    institution: "Stanford University School of Medicine",
    study: "Long-term Outcomes of T2D Management with GLP-1 Agonists",
    status: "pending",
    sent: "Mar 5, 2025",
    compensation: "$150 gift card",
    duration: "12 weeks",
    dataRequested: ["Conditions", "Medications", "Lab Results"],
    desc: "We are conducting a study on the effectiveness of GLP-1 agonist medications for Type 2 Diabetes management. Your profile matches our criteria.",
  },
  {
    institution: "Mayo Clinic Research",
    study: "Hypertension and Cardiovascular Risk in T2D Patients",
    status: "accepted",
    sent: "Feb 12, 2025",
    compensation: "$200 + follow-up benefits",
    duration: "6 months",
    dataRequested: ["Conditions", "Medications", "Vitals", "Wearable Data"],
    desc: "Research on the relationship between hypertension management and cardiovascular outcomes in patients with Type 2 Diabetes.",
  },
  {
    institution: "NIH National Heart Study",
    study: "ACE Inhibitor Effectiveness Cohort",
    status: "declined",
    sent: "Jan 8, 2025",
    compensation: "$100",
    duration: "8 weeks",
    dataRequested: ["Medications", "Lab Results"],
    desc: "Evaluating the effectiveness of ACE inhibitors in diverse patient populations.",
  },
];

const statusStyles: Record<string, string> = {
  pending: "status-pending",
  accepted: "status-active",
  declined: "bg-red-50 text-red-700 border border-red-200 inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium",
};

export default function Invitations() {
  return (
    <AppLayout title="Research Invitations">
      <div className="p-6 space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-1">Research Invitations</h2>
          <p className="text-muted-foreground text-sm">Institutions can invite you to participate in studies based on your de-identified profile. You control all consent.</p>
        </div>

        <div className="space-y-4">
          {invitations.map((inv, i) => (
            <div key={i} className="vault-card p-6">
              <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4 mb-4">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-xl bg-accent flex items-center justify-center flex-shrink-0">
                    <FlaskConical className="w-6 h-6 text-accent-foreground" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <span className={statusStyles[inv.status]}>
                        {inv.status === "pending" && "Pending Review"}
                        {inv.status === "accepted" && <><CheckCircle2 className="w-3 h-3" /> Accepted</>}
                        {inv.status === "declined" && <><X className="w-3 h-3" /> Declined</>}
                      </span>
                    </div>
                    <h3 className="font-semibold text-foreground">{inv.study}</h3>
                    <div className="text-sm text-muted-foreground flex items-center gap-1.5 mt-0.5">
                      <Building2 className="w-3.5 h-3.5" /> {inv.institution}
                    </div>
                  </div>
                </div>

                {inv.status === "pending" && (
                  <div className="flex gap-2 flex-shrink-0">
                    <Button variant="outline" size="sm" className="gap-1.5">
                      <X className="w-3.5 h-3.5" /> Decline
                    </Button>
                    <Button size="sm" className="gap-1.5 bg-primary text-primary-foreground hover:opacity-90 shadow-teal">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Accept & Grant Consent
                    </Button>
                  </div>
                )}
              </div>

              <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{inv.desc}</p>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                <div className="bg-muted/50 rounded-lg px-3 py-2">
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-0.5">Duration</div>
                  <div className="text-sm font-medium text-foreground">{inv.duration}</div>
                </div>
                <div className="bg-muted/50 rounded-lg px-3 py-2">
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-0.5">Compensation</div>
                  <div className="text-sm font-medium text-foreground">{inv.compensation}</div>
                </div>
                <div className="bg-muted/50 rounded-lg px-3 py-2">
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-0.5">Sent</div>
                  <div className="text-sm font-medium text-foreground flex items-center gap-1"><Calendar className="w-3 h-3" /> {inv.sent}</div>
                </div>
                <div className="bg-muted/50 rounded-lg px-3 py-2">
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-0.5">Data Requested</div>
                  <div className="text-xs text-foreground flex flex-wrap gap-1">
                    {inv.dataRequested.map((d) => (
                      <span key={d} className="bg-background px-1.5 py-0.5 rounded border border-border text-[10px]">{d}</span>
                    ))}
                  </div>
                </div>
              </div>

              <Button variant="ghost" size="sm" className="text-xs gap-1 -ml-2">
                <FileText className="w-3 h-3" /> View Full Study Details <ArrowRight className="w-3 h-3" />
              </Button>
            </div>
          ))}
        </div>
      </div>
    </AppLayout>
  );
}
