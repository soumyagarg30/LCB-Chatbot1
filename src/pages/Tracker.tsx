import { useEffect, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { CheckCircle2, Circle, Clock3, ListChecks } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getMarketingPlans, MarketingPlan, PlanStatus, updateMarketingPlan } from "@/utils/api";
import { toast } from "sonner";

const statuses: { value: PlanStatus; label: string; icon: typeof Circle; className: string }[] = [
  { value: "not_started", label: "Not started", icon: Circle, className: "text-slate-400" },
  { value: "in_progress", label: "In progress", icon: Clock3, className: "text-amber-500" },
  { value: "complete", label: "Complete", icon: CheckCircle2, className: "text-emerald-500" },
];

const Tracker = () => {
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("lcb_user") || "{}") as { role?: string };
  const [plans, setPlans] = useState<MarketingPlan[]>([]); const [loading, setLoading] = useState(true);
  useEffect(() => { getMarketingPlans().then(setPlans).catch((e) => toast.error(e.message)).finally(() => setLoading(false)); }, []);
  if (user.role !== "admin") return <Navigate to="/marketing" replace />;
  const changeStatus = async (plan: MarketingPlan, status: PlanStatus) => { try { const updated = await updateMarketingPlan(plan.id, status); setPlans((all) => all.map((item) => item.id === plan.id ? updated : item)); } catch (e) { toast.error(e instanceof Error ? e.message : "Could not update plan."); } };
  const complete = plans.filter((plan) => plan.status === "complete").length;
  const logout = () => { localStorage.removeItem("lcb_auth_token"); localStorage.removeItem("lcb_user"); navigate("/login", { replace: true }); };
  return <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-slate-950 to-slate-900 text-white"><header className="mx-auto max-w-7xl px-6 pt-6"><nav className="flex items-center justify-between rounded-full border border-white/10 bg-white/5 px-4 py-3"><div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-2xl bg-emerald-400/15">🌿</div><span className="font-semibold">LCB Growth Hub</span></div><div className="flex items-center gap-4 text-sm"><Link to="/" className="text-slate-300">Chat</Link><Link to="/marketing" className="text-slate-300">Marketing</Link><Link to="/tracker" className="text-emerald-300">Admin tracker</Link><Button size="sm" variant="outline" onClick={logout} className="border-white/20 bg-white/10 text-white">Logout</Button></div></nav></header><main className="mx-auto max-w-7xl px-6 py-9"><div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end"><div><p className="text-xs font-semibold uppercase tracking-[.2em] text-emerald-300">Admin only</p><h1 className="mt-2 text-3xl font-semibold">Implementation tracker</h1><p className="mt-2 text-slate-300">Every saved marketing plan, with a clear execution checklist.</p></div><div className="rounded-2xl border border-emerald-300/20 bg-emerald-300/10 px-5 py-3"><span className="text-2xl font-semibold">{complete}/{plans.length}</span><span className="ml-2 text-sm text-emerald-100">plans complete</span></div></div><div className="mt-8 space-y-4">{loading ? <p className="text-slate-300">Loading plans…</p> : plans.length === 0 ? <div className="rounded-3xl border border-dashed border-white/20 p-10 text-center text-slate-300"><ListChecks className="mx-auto mb-3 text-emerald-300"/>Plans saved from the Marketing tab will appear here.</div> : plans.map((plan) => { const selected = statuses.find((status) => status.value === plan.status)!; const Icon = selected.icon; return <article key={plan.id} className="rounded-3xl border border-white/10 bg-white/5 p-6"><div className="flex flex-col justify-between gap-4 lg:flex-row"><div className="max-w-3xl"><div className="flex items-center gap-2"><Icon size={19} className={selected.className}/><h2 className="text-lg font-semibold">{plan.title}</h2></div><p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-slate-300">{plan.strategy}</p><p className="mt-4 text-xs text-slate-500">Owner: {plan.owner || "Unassigned"} · Created {new Date(plan.created_at).toLocaleDateString()}</p></div><div className="flex min-w-40 flex-row gap-2 lg:flex-col">{statuses.map((status) => <button key={status.value} onClick={() => changeStatus(plan, status.value)} className={`rounded-xl border px-3 py-2 text-left text-xs font-medium transition ${plan.status === status.value ? "border-emerald-300 bg-emerald-300/15 text-emerald-100" : "border-white/10 text-slate-400 hover:bg-white/5"}`}>{status.label}</button>)}</div></div></article>; })}</div></main></div>;
};
export default Tracker;
