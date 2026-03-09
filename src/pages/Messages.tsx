import { useState } from "react";
import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MessageSquare, Send, Search, User, ChevronRight, DollarSign } from "lucide-react";

const conversations = [
  {
    id: "anon_9341",
    lastMessage: "Yes, Metformin has worked well for me. I've been on it for 3 years...",
    unread: true,
    time: "2h ago",
    status: "active",
  },
  {
    id: "anon_1827",
    lastMessage: "Thanks for reaching out! Happy to share my experience with managing T2D and GERD together.",
    unread: false,
    time: "1d ago",
    status: "active",
  },
  {
    id: "anon_6482",
    lastMessage: "The combination of Lisinopril and Atorvastatin has been great for my numbers.",
    unread: false,
    time: "3d ago",
    status: "active",
  },
];

const pendingRequests = [
  { id: "anon_8712", feeAmount: "0.002 ETH", time: "5h ago" },
];

export default function Messages() {
  const [selectedConvo, setSelectedConvo] = useState(conversations[0]);
  const [message, setMessage] = useState("");

  const mockMessages = [
    { from: "anon_9341", text: "Hi! I noticed we have similar health profiles. I was curious about your experience with Metformin?", time: "3d ago", self: false },
    { from: "me", text: "Hey! Yes happy to share. I've been on 500mg twice daily for about 2 years now. It's helped a lot with glucose control.", time: "3d ago", self: true },
    { from: "anon_9341", text: "That's great to hear! Did you experience any side effects initially?", time: "2d ago", self: false },
    { from: "me", text: "Some GI issues in the first couple weeks, but they resolved. Taking it with food helped.", time: "2d ago", self: true },
    { from: "anon_9341", text: "Yes, Metformin has worked well for me. I've been on it for 3 years now. The key was being consistent and pairing it with diet changes.", time: "2h ago", self: false },
  ];

  return (
    <AppLayout title="Messages">
      <div className="flex h-[calc(100vh-64px)]">
        {/* Sidebar */}
        <div className="w-80 border-r border-border flex flex-col bg-card flex-shrink-0">
          <div className="p-4 border-b border-border">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input placeholder="Search messages..." className="pl-9 h-9" />
            </div>
          </div>

          {pendingRequests.length > 0 && (
            <div className="p-4 border-b border-border">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Pending Requests</div>
              {pendingRequests.map((req) => (
                <div key={req.id} className="flex items-center justify-between p-2 rounded-lg bg-amber-50 border border-amber-200">
                  <div>
                    <div className="font-mono text-sm font-medium text-foreground">{req.id}</div>
                    <div className="text-xs text-muted-foreground flex items-center gap-1">
                      <DollarSign className="w-3 h-3" /> {req.feeAmount}
                    </div>
                  </div>
                  <div className="flex gap-1.5">
                    <Button size="sm" variant="ghost" className="h-7 px-2 text-xs">Decline</Button>
                    <Button size="sm" className="h-7 px-2 text-xs bg-primary text-primary-foreground">Accept</Button>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {conversations.map((convo) => (
              <button
                key={convo.id}
                className={`w-full flex items-start gap-3 p-4 text-left hover:bg-muted/50 transition-colors ${selectedConvo.id === convo.id ? "bg-muted" : ""}`}
                onClick={() => setSelectedConvo(convo)}
              >
                <div className="w-9 h-9 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-accent-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className={`font-mono text-sm ${convo.unread ? "font-semibold text-foreground" : "text-foreground"}`}>{convo.id}</span>
                    <span className="text-[10px] text-muted-foreground flex-shrink-0">{convo.time}</span>
                  </div>
                  <div className={`text-xs truncate mt-0.5 ${convo.unread ? "text-foreground" : "text-muted-foreground"}`}>{convo.lastMessage}</div>
                </div>
                {convo.unread && <div className="w-2 h-2 rounded-full bg-primary flex-shrink-0 mt-3" />}
              </button>
            ))}
          </div>
        </div>

        {/* Chat */}
        <div className="flex-1 flex flex-col">
          <div className="h-14 border-b border-border flex items-center justify-between px-5 bg-card flex-shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center">
                <User className="w-4 h-4 text-accent-foreground" />
              </div>
              <div>
                <div className="font-mono font-semibold text-sm text-foreground">{selectedConvo.id}</div>
                <div className="text-[10px] text-muted-foreground">Similar profile · T2D, Metformin</div>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="gap-1 text-xs">
              View Profile <ChevronRight className="w-3 h-3" />
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {mockMessages.map((msg, i) => (
              <div key={i} className={`flex ${msg.self ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-md rounded-2xl px-4 py-2.5 ${msg.self ? "bg-primary text-primary-foreground rounded-br-md" : "bg-muted text-foreground rounded-bl-md"}`}>
                  <div className="text-sm">{msg.text}</div>
                  <div className={`text-[10px] mt-1 ${msg.self ? "text-primary-foreground/60" : "text-muted-foreground"}`}>{msg.time}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="p-4 border-t border-border bg-card flex-shrink-0">
            <div className="flex items-center gap-2">
              <Input
                placeholder="Type a message..."
                className="flex-1"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
              />
              <Button className="gap-2 bg-primary text-primary-foreground hover:opacity-90 shadow-teal">
                <Send className="w-4 h-4" />
                Send
              </Button>
            </div>
            <div className="text-[10px] text-muted-foreground mt-2">Messages are end-to-end encrypted. Neither party sees real identities.</div>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
