import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowUp, Search, ShieldCheck } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import JudgeBadge from "@/components/JudgeBadge";
import { ChatJudgment, sendCompetitorMessage } from "@/utils/api";
import { toast } from "sonner";

type Message = { role: "user" | "assistant"; text: string; judgment?: ChatJudgment };

const suggestions = [
  "Assess our main competitors and identify the three best opportunities to get ahead",
  "Compare our positioning, dealer network, and farmer engagement with competitors",
  "Create a 30/60/90-day competitive action plan for LCB Fertilizers",
];

const CompetitorIntelligence = () => {
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("lcb_user") || "{}") as { role?: string };
  const [messages, setMessages] = useState<Message[]>([{
    role: "assistant",
    text: "I’m the LCB Competitive Intelligence Strategist. Give me competitor names, websites, reports, or a market question, and I’ll assess the evidence and recommend how LCB can get ahead.",
  }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const logout = () => {
    localStorage.removeItem("lcb_auth_token");
    localStorage.removeItem("lcb_user");
    navigate("/login", { replace: true });
  };

  const ask = async (text: string) => {
    if (!text.trim() || loading) return;
    setMessages((current) => [...current, { role: "user", text }]);
    setInput("");
    setLoading(true);
    try {
      const result = await sendCompetitorMessage(text);
      setMessages((current) => [...current, { role: "assistant", text: result.response, judgment: result.judgment }]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not reach the competitor intelligence agent.";
      toast.error(message);
      setMessages((current) => [...current, { role: "assistant", text: `I couldn’t complete the assessment. ${message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const submit = (event: FormEvent) => { event.preventDefault(); void ask(input); };

  return <div className="min-h-screen bg-gradient-to-br from-indigo-950 via-slate-950 to-emerald-950 text-white">
    <header className="mx-auto max-w-7xl px-6 pt-6">
      <nav className="flex flex-wrap items-center justify-between gap-4 rounded-3xl border border-white/10 bg-white/5 px-5 py-3 backdrop-blur-xl">
        <div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-2xl bg-indigo-400/15">🔭</div><span className="font-semibold">LCB Intelligence Hub</span></div>
        <div className="flex flex-wrap items-center gap-4 text-sm"><Link to="/" className="text-slate-300 hover:text-white">Chat</Link><Link to="/marketing" className="text-slate-300 hover:text-white">Marketing</Link><Link to="/competitors" className="text-indigo-300">Competitors</Link>{user.role === "admin" && <><Link to="/agent-manager" className="text-slate-300 hover:text-white">Agent Manager</Link><Link to="/tracker" className="text-slate-300 hover:text-white">Admin tracker</Link></>}<Button size="sm" variant="outline" onClick={logout} className="border-white/20 bg-white/10 text-white hover:bg-white/20">Logout</Button></div>
      </nav>
    </header>
    <main className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside className="space-y-5">
        <div className="rounded-[2rem] border border-indigo-400/20 bg-indigo-400/10 p-6"><Search className="text-indigo-200"/><h1 className="mt-4 text-2xl font-semibold">Competitive Intelligence</h1><p className="mt-2 text-sm leading-6 text-slate-300">Assess competitors, expose strategic gaps, and turn evidence into prioritized actions for LCB Fertilizers.</p></div>
        <div className="rounded-[2rem] border border-white/10 bg-white/5 p-5"><div className="flex items-center gap-2 text-sm font-semibold"><ShieldCheck size={17} className="text-emerald-300"/> Evidence-first analysis</div><p className="mt-2 text-sm text-slate-400">The agent uses uploaded documents and website sources, labels assumptions, and will not invent competitor facts.</p></div>
        <div className="space-y-2">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => void ask(suggestion)} disabled={loading} className="w-full rounded-2xl border border-white/10 bg-white/5 p-4 text-left text-sm text-slate-200 hover:border-indigo-300/50 hover:bg-white/10 disabled:opacity-50">{suggestion}</button>)}</div>
      </aside>
      <section className="flex min-h-[700px] min-w-0 flex-col overflow-hidden rounded-[2rem] border border-white/10 bg-slate-50 shadow-2xl">
        <div className="bg-gradient-to-r from-indigo-600 to-emerald-600 px-6 py-5"><p className="text-xs font-semibold uppercase tracking-[.2em] text-indigo-100">Know the landscape</p><h2 className="mt-1 text-2xl font-semibold">Competitor strategy agent</h2></div>
        <div className="flex-1 space-y-5 overflow-y-auto p-6">{messages.map((message, index) => <div key={`${message.role}-${index}`} className={message.role === "user" ? "ml-auto max-w-[82%] rounded-3xl bg-indigo-600 px-5 py-4 text-white" : "max-w-[95%] rounded-3xl border border-slate-200 bg-white px-5 py-5 text-slate-800 shadow-sm"}>{message.role === "user" ? <p className="whitespace-pre-wrap text-sm leading-6">{message.text}</p> : <><div className="prose prose-slate max-w-none prose-headings:text-indigo-950 prose-p:text-sm prose-li:text-sm prose-table:block prose-table:overflow-x-auto"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown></div>{message.judgment && <JudgeBadge judgment={message.judgment}/>}</>}</div>)}{loading && <div className="max-w-[95%] rounded-3xl border border-indigo-100 bg-indigo-50 px-5 py-4 text-sm text-indigo-900">Assessing the competitive landscape…</div>}</div>
        <form onSubmit={submit} className="flex gap-3 border-t border-slate-200 bg-white p-5"><Input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about competitors, threats, gaps, or ways to get ahead…" className="rounded-full text-slate-900" disabled={loading}/><Button type="submit" disabled={loading || !input.trim()} className="rounded-full bg-indigo-600 hover:bg-indigo-700"><ArrowUp size={18}/></Button></form>
      </section>
    </main>
  </div>;
};

export default CompetitorIntelligence;
