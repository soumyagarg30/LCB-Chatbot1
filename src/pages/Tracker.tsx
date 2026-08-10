import { useEffect, useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { CheckCircle2, Circle, ListChecks, MessageCircleQuestion } from "lucide-react";
import { Button } from "@/components/ui/button";
import { askTrackerQuestion, ChatJudgment, getMarketingPlans, MarketingPlan, PlanStatus, updateMarketingPlan } from "@/utils/api";
import JudgeBadge from "@/components/JudgeBadge";
import { toast } from "sonner";

const Tracker = () => {
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("lcb_user") || "{}") as { role?: string };
  const [plans, setPlans] = useState<MarketingPlan[]>([]);
  const [loading, setLoading] = useState(true);
  const [assistantQuestion, setAssistantQuestion] = useState("");
  const [assistantAnswer, setAssistantAnswer] = useState("");
  const [assistantJudgment, setAssistantJudgment] = useState<ChatJudgment>();
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [expandedPlanId, setExpandedPlanId] = useState<number | null>(null);
  const [showStrategiesDropdown, setShowStrategiesDropdown] = useState(false);

  useEffect(() => {
    getMarketingPlans()
      .then(setPlans)
      .catch((e) => toast.error(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (user.role !== "admin") return <Navigate to="/marketing" replace />;

  const changeStatus = async (plan: MarketingPlan, status: PlanStatus) => {
    try {
      const updated = await updateMarketingPlan(plan.id, status);
      setPlans((all) => all.map((item) => item.id === plan.id ? updated : item));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not update plan.");
    }
  };

  const askAssistant = async (event: FormEvent) => {
    event.preventDefault();
    const trimmedQuestion = assistantQuestion.trim();
    if (!trimmedQuestion) return;

    setAssistantLoading(true);
    setAssistantJudgment(undefined);
    try {
      const response = await askTrackerQuestion(trimmedQuestion);
      if (!response.success) {
        toast.error(response.error || "Unable to answer that question.");
        setAssistantAnswer(response.error || "Unable to answer that question.");
      } else {
        setAssistantAnswer(response.response);
        setAssistantJudgment(response.judgment);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to answer that question.";
      setAssistantAnswer(message);
      toast.error(message);
    } finally {
      setAssistantLoading(false);
    }
  };

  const complete = plans.filter((plan) => plan.status === "complete").length;
  const logout = () => {
    localStorage.removeItem("lcb_auth_token");
    localStorage.removeItem("lcb_user");
    navigate("/login", { replace: true });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-slate-950 to-slate-900 text-white">
      <header className="mx-auto max-w-7xl px-6 pt-6">
        <nav className="flex items-center justify-between rounded-full border border-white/10 bg-white/5 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-emerald-400/15">🌿</div>
            <span className="font-semibold">LCB Growth Hub</span>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <Link to="/" className="text-slate-300">General Chat</Link>
            <Link to="/marketing" className="text-slate-300">Marketing</Link>
            <Link to="/agent-manager" className="text-slate-300">Agent Manager</Link>
            <Link to="/tracker" className="text-emerald-300">Admin tracker</Link>
            <Button size="sm" variant="outline" onClick={logout} className="border-white/20 bg-white/10 text-white">Logout</Button>
          </div>
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-9">
        <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[.2em] text-emerald-300">Admin only</p>
            <h1 className="mt-2 text-3xl font-semibold">Implementation tracker</h1>
            <p className="mt-2 text-slate-300">Every saved marketing plan, with a clear execution checklist.</p>
          </div>
          <div className="rounded-2xl border border-emerald-300/20 bg-emerald-300/10 px-5 py-3">
            <span className="text-2xl font-semibold">{complete}/{plans.length}</span>
            <span className="ml-2 text-sm text-emerald-100">plans complete</span>
          </div>
        </div>

        <div className="mt-8 space-y-4">
          <div className="rounded-3xl border border-emerald-300/20 bg-emerald-400/10 p-5">
            <div className="flex items-center gap-3">
              <MessageCircleQuestion className="text-emerald-300" />
              <div>
                <h2 className="text-lg font-semibold text-white">Tracker assistant</h2>
                <p className="text-sm text-emerald-100/90">Ask about saved strategies, owners, progress, and tracker details. The answers come from the SQLite tracker data.</p>
              </div>
            </div>
            <div className="mt-4 space-y-3">
              <textarea
                value={assistantQuestion}
                onChange={(event) => setAssistantQuestion(event.target.value)}
                rows={3}
                className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 text-sm text-slate-100 outline-none placeholder:text-slate-500"
                placeholder="Example: What strategies are in the tracker?"
              />
              <div className="flex flex-wrap items-center justify-between gap-3">
                <Button type="button" onClick={() => void askAssistant({ preventDefault: () => undefined } as FormEvent)} disabled={assistantLoading} className="bg-emerald-500 text-black hover:bg-emerald-400">
                  {assistantLoading ? "Thinking…" : "Ask assistant"}
                </Button>
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowStrategiesDropdown(!showStrategiesDropdown)}
                    className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs text-emerald-100/80 transition hover:bg-white/10"
                  >
                    <CheckCircle2 size={14} className="text-emerald-400" />
                    <span>Tick strategies</span>
                  </button>
                  {showStrategiesDropdown && (
                    <div className="absolute right-0 top-full mt-2 w-72 rounded-2xl border border-white/10 bg-slate-950 shadow-xl">
                      <div className="max-h-96 space-y-1 overflow-y-auto p-3">
                        {plans.length === 0 ? (
                          <p className="px-3 py-2 text-xs text-slate-400">No strategies yet</p>
                        ) : (
                          plans.map((plan) => {
                            const isComplete = plan.status === "complete";
                            return (
                              <button
                                key={plan.id}
                                type="button"
                                onClick={() => void changeStatus(plan, isComplete ? "not_started" : "complete")}
                                className="flex w-full items-center gap-2 rounded-lg border border-transparent px-3 py-2 text-left text-xs text-slate-100 transition hover:bg-white/5 hover:border-white/10"
                              >
                                {isComplete ? (
                                  <CheckCircle2 size={16} className="text-emerald-400" />
                                ) : (
                                  <Circle size={16} className="text-slate-400" />
                                )}
                                <div className="flex-1 truncate">
                                  <p className="truncate font-medium">{plan.title}</p>
                                  <p className="text-[10px] text-slate-500 line-clamp-1">{plan.owner || "Unassigned"}</p>
                                </div>
                              </button>
                            );
                          })
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            {assistantAnswer && (
              <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm leading-7 text-slate-200 whitespace-pre-wrap">
                {assistantAnswer}
                <JudgeBadge judgment={assistantJudgment} />
              </div>
            )}
          </div>

          {loading ? (
            <p className="text-slate-300">Loading plans…</p>
          ) : plans.length === 0 ? (
            <div className="rounded-3xl border border-dashed border-white/20 p-10 text-center text-slate-300">
              <ListChecks className="mx-auto mb-3 text-emerald-300" />
              Plans saved from the Marketing tab will appear here.
            </div>
          ) : (
            plans.map((plan) => {
              const isComplete = plan.status === "complete";
              const isExpanded = expandedPlanId === plan.id;
              return (
                <article key={plan.id} className="overflow-hidden rounded-3xl border border-white/10 bg-white/5">
                  <button
                    type="button"
                    onClick={() => setExpandedPlanId(isExpanded ? null : plan.id)}
                    className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
                  >
                    <div className="flex items-center gap-2">
                      {isComplete ? (
                        <CheckCircle2 size={19} className="text-emerald-500" />
                      ) : (
                        <Circle size={19} className="text-slate-400" />
                      )}
                      <div>
                        <h2 className="text-base font-semibold text-white">{plan.title}</h2>
                        <p className="mt-1 text-sm text-slate-400">{plan.strategy.slice(0, 100)}{plan.strategy.length > 100 ? "..." : ""}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] uppercase tracking-[.2em] text-slate-300">
                        {isComplete ? "Done" : "Open"}
                      </span>
                    </div>
                  </button>

                  {isExpanded && (
                    <div className="border-t border-white/10 bg-slate-950/20 px-5 py-4">
                      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                        <div className="max-w-3xl">
                          <p className="whitespace-pre-wrap text-sm leading-6 text-slate-300">{plan.strategy}</p>
                          <p className="mt-4 text-xs text-slate-500">Owner: {plan.owner || "Unassigned"} · Created {new Date(plan.created_at).toLocaleDateString()}</p>
                        </div>
                        <div className="flex items-center">
                          <button
                            type="button"
                            aria-label={isComplete ? "Mark plan as not complete" : "Mark plan as complete"}
                            onClick={() => void changeStatus(plan, isComplete ? "not_started" : "complete")}
                            className={`rounded-full border p-2 transition ${isComplete ? "border-emerald-400 bg-emerald-400/15" : "border-white/10 bg-white/5 hover:bg-white/10"}`}
                          >
                            {isComplete ? (
                              <CheckCircle2 size={18} className="text-emerald-400" />
                            ) : (
                              <Circle size={18} className="text-slate-400" />
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </article>
              );
            })
          )}
        </div>
      </main>
    </div>
  );
};

export default Tracker;
