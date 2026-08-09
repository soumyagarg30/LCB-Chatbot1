import type { ChatJudgment } from "@/utils/api";

const JudgeBadge = ({ judgment }: { judgment?: ChatJudgment }) => {
  const completed = judgment?.status === "completed" || judgment?.status === "fallback";
  // A skipped judge is internal metadata, not a user-facing error. Only show
  // the badge when an evaluation actually produced a numeric score.
  if (!completed || typeof judgment?.score !== "number") return null;
  const styles = completed
    ? judgment.verdict === "pass"
      ? "border-emerald-300/60 bg-emerald-100 text-emerald-800"
      : judgment.verdict === "warning"
        ? "border-amber-300/60 bg-amber-100 text-amber-800"
        : "border-red-300/60 bg-red-100 text-red-800"
    : "border-slate-300 bg-slate-100 text-slate-600";
  const detail = judgment?.feedback || judgment?.error ||
    "No judgment was returned. Restart the backend and try again.";
  const dimensionLabels: Record<string, string> = {
    relevance: "Relevance", correctness: "Correctness", groundedness: "Grounding",
    clarity: "Clarity", safety: "Safety",
  };

  return <div className={`mt-3 rounded-xl border px-3 py-2 text-xs ${styles}`} title={detail}>
    <div className="flex flex-wrap items-center justify-between gap-2 font-semibold">
      <span>{judgment?.status === "fallback" ? "Fallback Judge" : "LLM Judge"}: {completed ? judgment.verdict : "not judged"}</span>
      <span>{completed ? `${judgment.score}/100` : judgment?.status || "not received"}</span>
    </div>
    {completed && judgment.dimensions && <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 font-normal sm:grid-cols-5">
      {Object.entries(judgment.dimensions).map(([name, score]) => <span key={name}>
        {dimensionLabels[name] || name}: <strong>{score}</strong>
      </span>)}
    </div>}
    {judgment?.missing_information && judgment.missing_information.length > 0 && <div className="mt-2">
      <p className="font-semibold">What is missing</p>
      <ul className="ml-4 mt-1 list-disc space-y-1 font-normal">
        {judgment.missing_information.map((item, index) => <li key={index}>{item}</li>)}
      </ul>
    </div>}
    {judgment?.issues && judgment.issues.length > 0 && <div className="mt-2">
      <p className="font-semibold">Problems found</p>
      <ul className="ml-4 mt-1 list-disc space-y-1 font-normal">
        {judgment.issues.map((issue, index) => <li key={index}>{issue}</li>)}
      </ul>
    </div>}
    <p className="mt-2 font-normal opacity-90"><strong>How to improve:</strong> {detail}</p>
  </div>;
};

export default JudgeBadge;
