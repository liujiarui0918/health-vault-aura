import PublicLayout from "@/components/PublicLayout";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { FlaskConical, Users, BarChart3, Shield, ArrowRight, CheckCircle2 } from "lucide-react";

const benefits = [
  {
    icon: Users,
    title: "Access a consented cohort",
    desc: "Search thousands of pseudonymous health profiles by condition, medication, age range, and more.",
  },
  {
    icon: FlaskConical,
    title: "Run ethical research campaigns",
    desc: "Create campaigns with clear scope, consent language, and compensation. Participants opt in on-chain.",
  },
  {
    icon: BarChart3,
    title: "Rich, de-identified data",
    desc: "Get structured health data with conditions, medications, wearable metrics — all verified and de-identified.",
  },
  {
    icon: Shield,
    title: "Compliance built in",
    desc: "Every consent is on-chain and auditable. Exportable reports for IRB and compliance requirements.",
  },
];

export default function InstitutionInfo() {
  return (
    <PublicLayout>
      <div className="py-24 container mx-auto px-6">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent text-accent-foreground text-xs font-medium mb-4">
            For Research Institutions
          </div>
          <h1 className="text-5xl font-bold text-foreground mb-4">Ethical health research, simplified</h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Access a consented, de-identified health data network. Run research campaigns with built-in compliance and participant consent management.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center mt-8">
            <Link to="/institution/dashboard">
              <Button size="lg" className="bg-primary text-primary-foreground hover:opacity-90 shadow-teal gap-2">
                Institution Portal <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
            <Link to="/pricing">
              <Button size="lg" variant="outline">View Pricing</Button>
            </Link>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          {benefits.map((b) => (
            <div key={b.title} className="vault-card-hover p-8">
              <div className="w-10 h-10 rounded-xl bg-accent flex items-center justify-center mb-4">
                <b.icon className="w-5 h-5 text-accent-foreground" />
              </div>
              <h3 className="font-semibold text-foreground mb-2">{b.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{b.desc}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 vault-card p-8 max-w-2xl mx-auto">
          <h3 className="font-semibold text-foreground text-xl mb-4">What's included in the Research plan</h3>
          <div className="grid sm:grid-cols-2 gap-2">
            {[
              "Verified institution badge",
              "Cohort search with 20+ filters",
              "Campaign creation & management",
              "Bulk participant invitations",
              "On-chain consent verification",
              "IRB export reports",
              "API access for data integration",
              "Dedicated compliance support",
            ].map((item) => (
              <div key={item} className="flex items-center gap-2 text-sm text-foreground">
                <CheckCircle2 className="w-4 h-4 text-primary flex-shrink-0" />
                {item}
              </div>
            ))}
          </div>
        </div>
      </div>
    </PublicLayout>
  );
}
