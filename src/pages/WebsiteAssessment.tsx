import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { assessWebsite } from "@/utils/api";
import { toast } from "sonner";

const WebsiteAssessment = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("lcb_user") || "{}") as { role?: string };
  const [websiteUrl, setWebsiteUrl] = useState("");
  const [isAssessing, setIsAssessing] = useState(false);
  const [assessment, setAssessment] = useState<string | null>(null);

  const handleLogout = () => {
    localStorage.removeItem("lcb_auth_token");
    localStorage.removeItem("lcb_user");
    navigate("/login", { replace: true });
  };

  const handleAssess = async () => {
    if (!websiteUrl.trim()) {
      toast.error("Enter a website URL to assess.");
      return;
    }

    try {
      setIsAssessing(true);
      setAssessment(null);
      const result = await assessWebsite(websiteUrl.trim());
      if (result.success) {
        setAssessment(result.assessment || "No assessment text returned.");
      } else {
        toast.error(result.error || "Website assessment failed.");
      }
    } catch (error) {
      console.error(error);
      toast.error("Website assessment failed.");
    } finally {
      setIsAssessing(false);
    }
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
              {user.role === "admin" && (
                <Link
                  to="/tracker"
                  className={`transition ${
                    location.pathname === "/tracker"
                      ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4"
                      : "text-slate-200 hover:text-white"
                  }`}
                >
                  Admin tracker
                </Link>
              )}
              <Button variant="outline" size="sm" className="border-white/20 bg-white/10 text-white hover:bg-white/20" onClick={handleLogout}>
                Logout
              </Button>
            </div>
          </nav>
        </header>

        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="rounded-[2rem] border border-white/10 bg-slate-950/80 p-8 shadow-[0_40px_120px_-48px_rgba(16,185,129,0.65)] backdrop-blur-xl">
            <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
              <div className="space-y-6">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-emerald-300/80">Website assessment</p>
                  <h1 className="mt-3 text-4xl font-semibold text-white">Assess a website instantly</h1>
                  <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-300">
                    Paste any URL and get a concise quality assessment of the page content. This is useful for deciding whether a page should be ingested into the knowledge base or for quick content validation.
                  </p>
                </div>

                <div className="space-y-4 rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                  <Input
                    value={websiteUrl}
                    onChange={(e) => setWebsiteUrl(e.target.value)}
                    placeholder="https://example.com"
                    className="w-full bg-slate-950/70 text-white rounded-2xl border border-white/10"
                    style={{ borderColor: "rgba(255, 255, 255, 0.08)" }}
                  />
                  <Button
                    onClick={handleAssess}
                    disabled={isAssessing || !websiteUrl.trim()}
                    className="w-full rounded-full bg-emerald-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-500/20 hover:bg-emerald-500"
                  >
                    {isAssessing ? "Assessing..." : "Run Website Assessment"}
                  </Button>
                </div>

                {assessment && (
                  <div className="rounded-[1.75rem] border border-emerald-200/30 bg-emerald-50/60 p-6 text-slate-900">
                    <h2 className="text-lg font-semibold text-emerald-900">Assessment result</h2>
                    <p className="mt-3 whitespace-pre-line text-sm leading-6">{assessment}</p>
                  </div>
                )}
              </div>

              <div className="space-y-6">
                <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                  <h2 className="text-lg font-semibold text-white">How this works</h2>
                  <ul className="mt-4 space-y-3 text-sm text-slate-300">
                    <li>• Fetch the website page and extract readable text.</li>
                    <li>• Send the extracted text to the LLM for an objective assessment.</li>
                    <li>• Get quality, trustworthiness, and knowledge-base suitability feedback.</li>
                    <li>• Use it before ingesting content or checking page relevance.</li>
                  </ul>
                </div>
                <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                  <h2 className="text-lg font-semibold text-white">Tips</h2>
                  <p className="mt-3 text-sm leading-6 text-slate-300">
                    Use the assessment tab for any landing page, product page, blog article, or support page you want to evaluate before adding it as knowledge.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default WebsiteAssessment;
