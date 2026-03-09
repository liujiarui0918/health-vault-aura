import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

// Public pages
import Landing from "./pages/Landing";
import PrivacyExplainer from "./pages/PrivacyExplainer";
import Pricing from "./pages/Pricing";
import FAQ from "./pages/FAQ";
import InstitutionInfo from "./pages/InstitutionInfo";

// Auth
import Login from "./pages/Login";
import Signup from "./pages/Signup";

// User pages
import Dashboard from "./pages/Dashboard";
import HealthDataUpload from "./pages/HealthDataUpload";
import Timeline from "./pages/Timeline";
import SearchProfiles from "./pages/SearchProfiles";
import Messages from "./pages/Messages";
import Invitations from "./pages/Invitations";
import ConsentCenter from "./pages/ConsentCenter";

// Institution pages
import InstitutionDashboard from "./pages/institution/InstitutionDashboard";
import CampaignManagement from "./pages/institution/CampaignManagement";

// Admin pages
import AdminDashboard from "./pages/admin/AdminDashboard";
import AuditLogs from "./pages/admin/AuditLogs";

import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Public */}
          <Route path="/" element={<Landing />} />
          <Route path="/privacy" element={<PrivacyExplainer />} />
          <Route path="/pricing" element={<Pricing />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/institutions" element={<InstitutionInfo />} />

          {/* Auth */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />

          {/* User App */}
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/health-data" element={<HealthDataUpload />} />
          <Route path="/timeline" element={<Timeline />} />
          <Route path="/search" element={<SearchProfiles />} />
          <Route path="/messages" element={<Messages />} />
          <Route path="/invitations" element={<Invitations />} />
          <Route path="/consent" element={<ConsentCenter />} />

          {/* Institution */}
          <Route path="/institution/dashboard" element={<InstitutionDashboard />} />
          <Route path="/institution/campaigns" element={<CampaignManagement />} />

          {/* Admin */}
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="/admin/audit" element={<AuditLogs />} />

          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
