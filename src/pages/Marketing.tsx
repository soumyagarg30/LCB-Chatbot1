import { FormEvent, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ArrowUp, BookOpen, Clipboard, Link2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatJudgment, ingestKnowledge, saveMarketingPlan, sendMarketingMessage } from "@/utils/api";
import JudgeBadge from "@/components/JudgeBadge";
import { toast } from "sonner";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = { role: "user" | "assistant"; text: string; judgment?: ChatJudgment; judgeExpected?: boolean };

const prompts = [
  "Create a 90-day growth plan for Navyakosh in our priority markets",
  "Build a dealer activation strategy for the next sowing season",
  "Suggest a farmer education campaign using the company knowledge base",
];

const Marketing = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("lcb_user") || "{}") as { role?: string; name?: string };
  const [messages, setMessages] = useState<Message[]>([{ role: "assistant", text: "I’m your LCB Marketing Strategist. I use the shared knowledge base to turn company information into practical marketing plans. What would you like to grow?" }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState<number | null>(null);
  const [websiteUrls, setWebsiteUrls] = useState("");
  const [addingSources, setAddingSources] = useState(false);

  const logout = () => { localStorage.removeItem("lcb_auth_token"); localStorage.removeItem("lcb_user"); navigate("/login", { replace: true }); };
  const ask = async (text: string) => {
    if (!text.trim() || loading) return;
    setMessages((current) => [...current, { role: "user", text }]); setInput(""); setLoading(true);
    try {
      const result = await sendMarketingMessage(text);
      setMessages((current) => [...current, { role: "assistant", text: result.response || "I could not generate a strategy for that request. Please try again.", judgment: result.judgment, judgeExpected: true }]);
    }
    catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Could not reach the marketing agent.";
      setMessages((current) => [...current, { role: "assistant", text: `I couldn’t generate the plan right now. ${errorMessage}` }]);
      toast.error(errorMessage);
    }
    finally { setLoading(false); }
  };
  const submit = (event: FormEvent) => { event.preventDefault(); ask(input); };
  const savePlan = async (index: number, strategy: string) => {
    const firstLine = strategy.split("\n").find((line) => line.trim())?.replace(/^#+\s*/, "").slice(0, 90) || "LCB marketing plan";
    setSaving(index);
    try { await saveMarketingPlan(firstLine, strategy); toast.success("Plan added to the admin implementation tracker."); }
    catch (error) { toast.error(error instanceof Error ? error.message : "Could not save plan."); }
    finally { setSaving(null); }
  };
  const addWebsiteSources = async (event: FormEvent) => {
    event.preventDefault();
    const urls = websiteUrls.split(/[\n,]/).map((url) => url.trim()).filter(Boolean);
    if (!urls.length) return toast.error("Enter at least one website URL.");
    if (urls.some((url) => !/^https?:\/\//i.test(url))) return toast.error("Each source must start with http:// or https://");
    setAddingSources(true);
    try {
      const result = await ingestKnowledge(urls, []);
      if (!result.success) throw new Error(result.error || "Could not add website sources.");
      setWebsiteUrls("");
      toast.success(`${result.count || urls.length} website source(s) added. The agent can now use them.`);
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not add website sources."); }
    finally { setAddingSources(false); }
  };

  return <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-slate-950 to-slate-900 text-white">
    <header className="mx-auto max-w-7xl px-6 pt-6"><nav className="flex items-center justify-between gap-4 rounded-full border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-xl"><div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-2xl bg-emerald-400/15">🌿</div><span className="font-semibold text-emerald-100">LCB Growth Hub</span></div><div className="flex items-center gap-4 text-sm"><Link to="/" className={location.pathname === "/" ? "text-white" : "text-slate-300 hover:text-white"}>Chat</Link><Link to="/marketing" className="text-emerald-300">Marketing</Link><Link to="/competitors" className="text-slate-300 hover:text-white">Competitors</Link>{user.role === "admin" && <><Link to="/agent-manager" className="text-slate-300 hover:text-white">Agent Manager</Link><Link to="/tracker" className="text-slate-300 hover:text-white">Admin tracker</Link></>}<Button size="sm" variant="outline" onClick={logout} className="border-white/20 bg-white/10 text-white hover:bg-white/20">Logout</Button></div></nav></header>
    <main className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside className="space-y-5"><div className="rounded-[2rem] border border-emerald-400/20 bg-emerald-400/10 p-6"><Sparkles className="text-emerald-200"/><h1 className="mt-4 text-2xl font-semibold">Marketing Strategist</h1><p className="mt-2 text-sm leading-6 text-slate-300">Creates growth plans grounded in the documents and links your admin has fed into the LCB knowledge hub.</p></div><div className="rounded-[2rem] border border-white/10 bg-white/5 p-5"><div className="flex items-center gap-2 text-sm font-semibold"><BookOpen size={17} className="text-emerald-300"/> Shared intelligence</div><p className="mt-2 text-sm text-slate-400">The agent retrieves relevant LCB information before proposing a plan.</p></div>{user.role === "admin" && <form onSubmit={addWebsiteSources} className="rounded-[2rem] border border-white/10 bg-white/5 p-5"><div className="flex items-center gap-2 text-sm font-semibold"><Link2 size={17} className="text-emerald-300"/> Add website sources</div><p className="mt-2 text-xs leading-5 text-slate-400">Paste one or more public page URLs, separated by commas or lines.</p><textarea value={websiteUrls} onChange={(event) => setWebsiteUrls(event.target.value)} placeholder="https://example.com/product" className="mt-3 min-h-24 w-full rounded-2xl border border-white/10 bg-slate-950/70 p-3 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-emerald-300" disabled={addingSources}/><Button type="submit" disabled={addingSources || !websiteUrls.trim()} className="mt-3 w-full rounded-full bg-emerald-600 hover:bg-emerald-700">{addingSources ? "Reading websites…" : "Add to agent knowledge"}</Button></form>}<div className="space-y-2">{prompts.map((prompt) => <button key={prompt} onClick={() => ask(prompt)} disabled={loading} className="w-full rounded-2xl border border-white/10 bg-white/5 p-4 text-left text-sm text-slate-200 transition hover:border-emerald-300/50 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50">{prompt}</button>)}</div></aside>
      <section className="flex min-h-[700px] min-w-0 flex-col overflow-hidden rounded-[2rem] border border-white/10 bg-slate-50 shadow-2xl"><div className="bg-gradient-to-r from-emerald-600 to-teal-500 px-6 py-5"><p className="text-xs font-semibold uppercase tracking-[.2em] text-emerald-100">Plan with confidence</p><h2 className="mt-1 text-2xl font-semibold">LCB marketing planning agent</h2></div><div className="flex-1 space-y-5 overflow-x-hidden overflow-y-auto p-6">{messages.map((message, index) => <div key={`${message.role}-${index}`} className={message.role === "user" ? "ml-auto max-w-[82%] break-words rounded-3xl bg-emerald-600 px-5 py-4 text-white" : "max-w-[95%] break-words rounded-3xl border border-slate-200 bg-white px-5 py-5 text-slate-800 shadow-sm"}>{message.role === "user" ? <p className="whitespace-pre-wrap text-sm leading-6">{message.text}</p> : <div className="prose prose-slate max-w-none break-words prose-headings:text-emerald-950 prose-h1:text-2xl prose-h2:mt-7 prose-h2:border-b prose-h2:border-emerald-100 prose-h2:pb-2 prose-h2:text-lg prose-h3:text-base prose-p:text-sm prose-p:leading-6 prose-li:text-sm prose-table:block prose-table:max-w-full prose-table:overflow-x-auto prose-th:bg-emerald-50 prose-th:px-3 prose-td:px-3 prose-td:py-2"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown></div>}{message.role === "assistant" && message.judgeExpected && <JudgeBadge judgment={message.judgment} />}{message.role === "assistant" && index > 0 && <Button onClick={() => savePlan(index, message.text)} disabled={saving === index} variant="outline" size="sm" className="mt-5 border-emerald-200 text-emerald-800 hover:bg-emerald-50"><Clipboard size={15} className="mr-2"/>{saving === index ? "Adding…" : "Add to implementation tracker"}</Button>}</div>)}{loading && <div className="max-w-[95%] rounded-3xl border border-emerald-100 bg-emerald-50 px-5 py-4 text-sm text-emerald-900 shadow-sm"><span className="font-semibold">Creating your strategy…</span><p className="mt-1 text-emerald-800/80">The reply will appear here in the chatbot.</p></div>}</div><form onSubmit={submit} className="flex gap-3 border-t border-slate-200 bg-white p-5"><Input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask for a campaign, growth strategy, or go-to-market plan…" className="rounded-full text-slate-900" disabled={loading}/><Button type="submit" disabled={loading || !input.trim()} className="rounded-full bg-emerald-600 hover:bg-emerald-700"><ArrowUp size={18}/></Button></form></section>
    </main>
  </div>;
};

export default Marketing;
