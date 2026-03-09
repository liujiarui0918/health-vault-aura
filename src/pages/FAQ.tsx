import PublicLayout from "@/components/PublicLayout";

const faqs = [
  {
    q: "Is my health data ever stored unencrypted?",
    a: "Never. All data is encrypted client-side before it leaves your device. We store only encrypted blobs and never have access to the raw content.",
  },
  {
    q: "What gets stored on the blockchain?",
    a: "Only encrypted metadata hashes, profile identifiers, consent grants/revocations, and audit anchors. No actual health data is ever on-chain.",
  },
  {
    q: "Can I delete my data?",
    a: "Yes. You can delete your encrypted data at any time from your settings. On-chain hashes are immutable by nature, but without the encryption key they are meaningless.",
  },
  {
    q: "What is de-identification?",
    a: "It's the process of removing or obscuring personal identifiers from your health data — things like your name, exact date of birth, address, etc. Our pipeline runs this automatically before any data is shared.",
  },
  {
    q: "How does private messaging work?",
    a: "You can send a private message to another user's profile by paying a small fee held in escrow. The recipient decides whether to accept. This prevents spam and ensures meaningful connections.",
  },
  {
    q: "Can institutions access my data without my consent?",
    a: "Absolutely not. Institutions can only see your de-identified, publicly searchable profile attributes. Direct contact or data sharing requires your explicit on-chain consent.",
  },
  {
    q: "Do I need a wallet to use HealthVault?",
    a: "A wallet is required for on-chain features (consent registry, message escrow, audit anchoring). Basic profile features work without one, but we recommend connecting a wallet for full functionality.",
  },
  {
    q: "What blockchains are supported?",
    a: "HealthVault currently uses Ethereum-compatible chains. We support WalletConnect and MetaMask for wallet connections.",
  },
];

export default function FAQ() {
  return (
    <PublicLayout>
      <div className="py-24 container mx-auto px-6 max-w-3xl">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent text-accent-foreground text-xs font-medium mb-4">
            FAQ
          </div>
          <h1 className="text-5xl font-bold text-foreground mb-4">Frequently asked questions</h1>
          <p className="text-xl text-muted-foreground">Everything you need to know about HealthVault's privacy model and features.</p>
        </div>

        <div className="space-y-4">
          {faqs.map((faq, i) => (
            <div key={i} className="vault-card p-6">
              <h3 className="font-semibold text-foreground mb-2">{faq.q}</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">{faq.a}</p>
            </div>
          ))}
        </div>
      </div>
    </PublicLayout>
  );
}
