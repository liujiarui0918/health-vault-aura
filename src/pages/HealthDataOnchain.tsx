import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import AppLayout from "@/components/AppLayout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { useWalletContext } from "@/contexts/WalletContext";
import { useToast } from "@/hooks/use-toast";
import {
  ArrowLeft,
  ArrowRight,
  Blocks,
  CheckCircle2,
  FileJson,
  Link2,
  Lock,
  Loader2,
  ShieldCheck,
  Sparkles,
  Wallet,
} from "lucide-react";

type OnchainOption = {
  id: string;
  title: string;
  description: string;
  visibility: string;
  payload: Record<string, unknown>;
};

type SimulationResult = {
  publishedItems: string[];
  protectedFields: string[];
  walletAddress: string;
  transactionHash: string;
  simulatedAt: string;
};

const ONCHAIN_OPTIONS: OnchainOption[] = [
  {
    id: "lab-summary",
    title: "Lab Summary Commitment",
    description: "A minimal summary proving this report contains blood chemistry and metabolic panel results.",
    visibility: "Only a summary hash and category labels would be displayed on-chain.",
    payload: {
      reportType: "Annual Blood Panel",
      categories: ["Lab Results", "Vitals"],
      abnormalCount: 2,
      confidence: "verified-preview",
    },
  },
  {
    id: "eligibility-signal",
    title: "Research Eligibility Signal",
    description: "A compact eligibility marker that can be shared with future matching flows.",
        visibility: "Only a boolean-style signal and consent timestamp would be visible on-chain.",
        payload: {
          signalType: "cardio-metabolic-interest",
          eligibility: true,
          consentWindow: "90 days",
          version: "release-v1",
        },
      },
  {
    id: "integrity-proof",
    title: "Report Integrity Proof",
    description: "A provenance record showing that the uploaded report was parsed and reviewed in HealthVault.",
    visibility: "Only document fingerprint metadata would be stored on-chain.",
        payload: {
          parser: "HealthVault AI Parser",
          fingerprint: "0x6f1d...c2a9",
          reviewedBy: "wallet-owner",
          environment: "production-preview",
        },
      },
];

const PROTECTED_FIELDS = [
  "Full name",
  "Date of birth",
  "Phone number",
  "Home address",
  "Hospital / clinic identifier",
  "Original PDF file",
];

const DEMO_WALLET_ADDRESS = "0x8F31A7c65d10B3fF2c9a6212e7A018A63A12F4D9";
const DEMO_TRANSACTION_HASH = "0x7c4d18b9a6ef10d0d99b221445af31dd2f91d4b6016f3da8937ebff201f0c2aa";

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

export default function HealthDataOnchain() {
  const { connect, address, shortAddress, isConnected, isConnecting, isWalletAvailable } = useWalletContext();
  const { toast } = useToast();
  const [selectedIds, setSelectedIds] = useState<string[]>(["lab-summary", "integrity-proof"]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);

  const selectedOptions = useMemo(
    () => ONCHAIN_OPTIONS.filter((option) => selectedIds.includes(option.id)),
    [selectedIds],
  );

  const toggleSelection = (id: string, checked: boolean) => {
    setSelectedIds((current) => {
      if (checked) {
        return current.includes(id) ? current : [...current, id];
      }
      return current.filter((item) => item !== id);
    });
  };

  const handleSimulateOnchain = async () => {
    if (selectedOptions.length === 0) {
      toast({
        title: "Select at least one item",
        description: "Choose one or more records before continuing with the on-chain flow.",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);

    try {
      let walletAddress = address;

      if (!walletAddress && isWalletAvailable) {
        walletAddress = await connect();
        if (!walletAddress) {
          toast({
            title: "Wallet connection was not completed",
            description: "Reconnect your browser wallet and try again.",
            variant: "destructive",
          });
          return;
        }
      }

      if (!walletAddress) {
        walletAddress = DEMO_WALLET_ADDRESS;
        toast({
          title: "Temporary wallet session",
          description: "No browser wallet was detected, so a temporary wallet address was prepared to continue this flow.",
        });
      }

      await wait(900);
      await wait(1200);

      setResult({
        publishedItems: selectedOptions.map((option) => option.title),
        protectedFields: PROTECTED_FIELDS,
        walletAddress,
        transactionHash: DEMO_TRANSACTION_HASH,
        simulatedAt: new Date().toLocaleString("en-US", {
          year: "numeric",
          month: "short",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        }),
      });

      toast({
        title: "On-chain record created",
        description: "Your transaction summary is now ready to review.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (result) {
    return (
      <AppLayout title="On-Chain Review">
        <div className="p-6">
          <div className="mx-auto max-w-4xl space-y-6">
            <section className="overflow-hidden rounded-3xl border border-border bg-[linear-gradient(135deg,hsl(171_72%_28%)_0%,hsl(191_63%_24%)_100%)] p-6 text-white shadow-[var(--shadow-lg)]">
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="space-y-2">
                  <Badge className="border-white/20 bg-white/10 text-white hover:bg-white/10">Transaction completed</Badge>
                  <h2 className="text-3xl font-bold tracking-tight">Selected health data has been prepared for on-chain publishing.</h2>
                  <p className="max-w-2xl text-sm text-white/80">
                    Review exactly what was published, what remained private, and which wallet session was used for confirmation.
                  </p>
                </div>
                <div className="rounded-2xl bg-white/10 px-4 py-3 text-right">
                  <div className="text-xs uppercase tracking-[0.18em] text-white/65">Wallet</div>
                  <div className="mt-1 text-sm font-semibold">{result.walletAddress}</div>
                </div>
              </div>
            </section>

            <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
              <div className="vault-card p-6">
                <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                  <CheckCircle2 className="h-4 w-4 text-primary" />
                  Published on-chain
                </h3>
                <div className="mt-4 space-y-3">
                  {result.publishedItems.map((item) => (
                    <div key={item} className="rounded-2xl border border-border bg-muted/20 p-4">
                      <div className="text-sm font-medium text-foreground">{item}</div>
                    </div>
                  ))}
                </div>

                <Separator className="my-5" />

                <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                  <Lock className="h-4 w-4 text-primary" />
                  Protected privacy fields
                </h3>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  {result.protectedFields.map((field) => (
                    <div key={field} className="rounded-2xl border border-emerald-100 bg-emerald-50/70 px-4 py-3 text-sm text-emerald-800">
                      {field}
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-6">
                <div className="vault-card p-6">
                  <h3 className="text-lg font-semibold text-foreground">Transaction receipt</h3>
                  <div className="mt-4 space-y-4 text-sm">
                    <div>
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Wallet address</div>
                      <div className="mt-1 font-medium text-foreground break-all">{result.walletAddress}</div>
                    </div>
                    <div>
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Transaction hash</div>
                      <div className="mt-1 font-medium text-foreground break-all">{result.transactionHash}</div>
                    </div>
                    <div>
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Recorded at</div>
                      <div className="mt-1 font-medium text-foreground">{result.simulatedAt}</div>
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-primary/20 bg-accent p-5">
                  <div className="flex items-start gap-3">
                    <ShieldCheck className="mt-0.5 h-4 w-4 text-primary" />
                    <div className="text-sm text-accent-foreground">
                      <strong>Privacy is still protected.</strong> Only summary-level commitments and integrity proofs are published. Raw report details and personal identifiers remain off-chain.
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button asChild className="gap-2">
                    <Link to="/health-data">
                      Back to report review
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                  </Button>
                  <Button
                    variant="outline"
                    className="gap-2"
                    onClick={() => {
                      setResult(null);
                    }}
                  >
                    Create another record
                  </Button>
                </div>
              </div>
            </section>
          </div>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout title="On-Chain Review">
      <div className="p-6">
        <div className="mx-auto max-w-6xl space-y-6">
          <section className="overflow-hidden rounded-3xl border border-border bg-[linear-gradient(135deg,hsl(220_45%_14%)_0%,hsl(193_60%_17%)_55%,hsl(171_62%_20%)_100%)] p-6 text-white shadow-[var(--shadow-lg)]">
            <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  <Badge className="border-white/20 bg-white/10 text-white hover:bg-white/10">Selective publishing</Badge>
                  <Badge className="border-white/20 bg-white/10 text-white hover:bg-white/10">Privacy protected</Badge>
                  <Badge className="border-white/20 bg-white/10 text-white hover:bg-white/10">Wallet confirmation</Badge>
                </div>
                <div>
                  <h2 className="text-3xl font-bold tracking-tight">Do you want to publish part of this report on-chain?</h2>
                  <p className="mt-2 max-w-2xl text-sm text-white/78">
                    Choose only the summary-level items you want to anchor. Privacy-sensitive fields stay protected and are never shown publicly on-chain.
                  </p>
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  <div className="rounded-2xl border border-white/10 bg-white/8 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-white/55">On-chain</div>
                    <div className="mt-2 text-sm font-semibold">Summary commitments</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/8 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-white/55">Off-chain</div>
                    <div className="mt-2 text-sm font-semibold">Personal identifiers and report file</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/8 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-white/55">Wallet</div>
                    <div className="mt-2 text-sm font-semibold">{isConnected ? shortAddress : "Connect on action"}</div>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/8 p-5">
                <div className="flex items-center gap-2 text-sm font-medium text-white/90">
                  <Sparkles className="h-4 w-4 text-teal-200" />
                  Privacy promise
                </div>
                <div className="mt-4 space-y-3">
                  {PROTECTED_FIELDS.slice(0, 4).map((field) => (
                    <div key={field} className="rounded-2xl border border-white/10 bg-white/6 px-4 py-3 text-sm text-white/82">
                      {field}
                    </div>
                  ))}
                </div>
                <div className="mt-5 rounded-2xl border border-emerald-300/15 bg-emerald-400/8 p-4 text-sm text-white/82">
                  Private health details are represented by hashed or summarized records only. The original data is not directly written to chain.
                </div>
              </div>
            </div>
          </section>

          <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <div className="vault-card p-6">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                    <Blocks className="h-4 w-4 text-primary" />
                    Select what goes on-chain
                  </h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Choose the summaries you want to publish while keeping private details protected.
                  </p>
                </div>
                <Badge variant="outline" className="border-primary/20 bg-accent/50 text-accent-foreground">
                  {selectedOptions.length} selected
                </Badge>
              </div>

              <div className="mt-5 space-y-4">
                {ONCHAIN_OPTIONS.map((option) => {
                  const checked = selectedIds.includes(option.id);

                  return (
                    <label
                      key={option.id}
                      className={`block rounded-3xl border p-5 transition-all ${checked ? "border-primary bg-accent/35 shadow-[var(--shadow-sm)]" : "border-border bg-background hover:border-primary/20"}`}
                    >
                      <div className="flex items-start gap-4">
                        <Checkbox
                          checked={checked}
                          onCheckedChange={(value) => toggleSelection(option.id, value === true)}
                          className="mt-1"
                        />
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <div className="text-base font-semibold text-foreground">{option.title}</div>
                            <Badge className="bg-slate-100 text-slate-700 hover:bg-slate-100">{checked ? "Selected" : "Optional"}</Badge>
                          </div>
                          <div className="mt-2 text-sm leading-6 text-muted-foreground">{option.description}</div>
                          <div className="mt-3 rounded-2xl border border-border bg-muted/25 px-4 py-3 text-xs text-muted-foreground">
                            {option.visibility}
                          </div>
                            <div className="mt-4 rounded-2xl bg-slate-950 p-4 text-xs leading-6 text-slate-100">
                            <div className="mb-2 flex items-center gap-2 text-slate-300">
                              <FileJson className="h-3.5 w-3.5" />
                              Payload preview
                            </div>
                            <pre className="overflow-auto">{JSON.stringify(option.payload, null, 2)}</pre>
                          </div>
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="space-y-6">
              <div className="vault-card p-6">
                <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                  <Lock className="h-4 w-4 text-primary" />
                  What stays private
                </h3>
                <div className="mt-4 space-y-3">
                  {PROTECTED_FIELDS.map((field) => (
                    <div key={field} className="rounded-2xl border border-emerald-100 bg-emerald-50/70 px-4 py-3 text-sm text-emerald-800">
                      {field}
                    </div>
                  ))}
                </div>
              </div>

              <div className="vault-card p-6">
                <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                  <Wallet className="h-4 w-4 text-primary" />
                  Wallet and execution
                </h3>
                <div className="mt-4 space-y-4 text-sm">
                  <div className="rounded-2xl border border-border bg-muted/20 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Wallet status</div>
                    <div className="mt-2 font-medium text-foreground">
                      {isConnected ? shortAddress : isWalletAvailable ? "Ready to connect on action" : "A temporary session address will be prepared if needed"}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-border bg-muted/20 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Execution model</div>
                    <div className="mt-2 font-medium text-foreground">Browser wallet confirmation</div>
                    <div className="mt-2 text-xs leading-5 text-muted-foreground">
                      The flow first attempts to connect the browser wallet. Once connected, it creates an on-chain record and returns a transaction receipt.
                    </div>
                  </div>
                </div>

                <div className="mt-5 flex flex-wrap gap-3">
                  <Button asChild variant="outline" className="gap-2">
                    <Link to="/health-data">
                      <ArrowLeft className="h-4 w-4" />
                      Back to results
                    </Link>
                  </Button>
                  <Button className="gap-2" disabled={isSubmitting || isConnecting} onClick={handleSimulateOnchain}>
                    {isSubmitting || isConnecting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Link2 className="h-4 w-4" />}
                    {isConnecting ? "Connecting wallet..." : isSubmitting ? "Publishing on-chain..." : "Write To Chain"}
                  </Button>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </AppLayout>
  );
}
