import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import Index from "./pages/Index";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import Marketing from "./pages/Marketing";
import Tracker from "./pages/Tracker";
import WebsiteAssessment from "./pages/WebsiteAssessment";
import AgentManager from "./pages/AgentManager";
import CompetitorIntelligence from "./pages/CompetitorIntelligence";

const ProtectedRoute = ({ children }: { children: JSX.Element }) => {
  const token = localStorage.getItem("lcb_auth_token");
  return token ? children : <Navigate to="/login" replace />;
};

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <HashRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<ProtectedRoute><Index /></ProtectedRoute>} />
          <Route path="/marketing" element={<ProtectedRoute><Marketing /></ProtectedRoute>} />
          <Route path="/competitors" element={<ProtectedRoute><CompetitorIntelligence /></ProtectedRoute>} />
          <Route path="/website-assessment" element={<ProtectedRoute><WebsiteAssessment /></ProtectedRoute>} />
          <Route path="/agent-manager" element={<ProtectedRoute><AgentManager /></ProtectedRoute>} />
          <Route path="/tracker" element={<ProtectedRoute><Tracker /></ProtectedRoute>} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </HashRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
