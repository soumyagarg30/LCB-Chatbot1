import { Link, useLocation, useNavigate } from "react-router-dom";
import ChatSection from "@/components/ChatSection";
import { Button } from "@/components/ui/button";

const Index = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("lcb_user") || "{}") as { role?: string };

  const handleLogout = () => {
    localStorage.removeItem("lcb_auth_token");
    localStorage.removeItem("lcb_user");
    navigate("/login", { replace: true });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-slate-950 to-slate-900 text-white overflow-hidden">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div
          className="absolute -top-36 -right-20 w-96 h-96 rounded-full bg-emerald-500 opacity-10 blur-3xl animate-blob"
          style={{ animationDuration: "10s" }}
        />
        <div
          className="absolute -bottom-40 -left-24 w-96 h-96 rounded-full bg-cyan-500 opacity-10 blur-3xl animate-blob animation-delay-2000"
          style={{ animationDuration: "12s" }}
        />
        <div
          className="absolute top-1/3 left-1/2 w-72 h-72 rounded-full bg-lime-400 opacity-8 blur-3xl animate-blob animation-delay-4000"
          style={{ animationDuration: "14s" }}
        />
      </div>

      <div className="relative z-10">
        <header className="max-w-7xl mx-auto px-6 pt-6">
          <nav className="flex items-center justify-between gap-6 rounded-full border border-white/10 bg-white/5 px-4 py-3 shadow-lg shadow-black/10 backdrop-blur-xl">
            <div className="flex items-center gap-3">
              <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-emerald-400/15 text-emerald-200 ring-1 ring-white/10">
                🌿
              </div>
              <span className="text-sm font-semibold tracking-wide text-emerald-100">LCB AI Assistant</span>
            </div>
            <div className="flex items-center gap-4 text-sm font-medium text-slate-100">
              <Link
                to="/"
                className={`transition ${
                  location.pathname === "/"
                    ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4"
                    : "text-slate-200 hover:text-white"
                }`}
              >
                Chat
              </Link>
              <Link
                to="/marketing"
                className={`transition ${
                  location.pathname === "/marketing"
                    ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4"
                    : "text-slate-200 hover:text-white"
                }`}
              >
                Marketing
              </Link>
              <Link
                to="/website-assessment"
                className={`transition ${
                  location.pathname === "/website-assessment"
                    ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4"
                    : "text-slate-200 hover:text-white"
                }`}
              >
                Assess Website
              </Link>
              {user.role === "admin" && <Link to="/tracker" className={`transition ${
                  location.pathname === "/tracker"
                    ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4"
                    : "text-slate-200 hover:text-white"
                }`}>Admin tracker</Link>}
              <Button variant="outline" size="sm" className="border-white/20 bg-white/10 text-white hover:bg-white/20" onClick={handleLogout}>
                Logout
              </Button>
            </div>
          </nav>
        </header>

        <main className="max-w-7xl mx-auto px-6 py-8">
          <ChatSection />
        </main>
      </div>
    </div>
  );
};

export default Index;
