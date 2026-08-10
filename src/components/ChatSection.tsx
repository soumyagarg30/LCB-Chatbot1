"use client";
import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ArrowRight, ChevronLeft, ChevronRight, Upload } from "lucide-react";
import MessageBubble from "./MessageBubble";
import { sendMessage, checkHealth, ingestKnowledge, ApiLanguage, ChatJudgment, KnowledgeSource } from "@/utils/api";
import { toast } from "sonner";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  judgment?: ChatJudgment;
  judgeExpected?: boolean;
  activeAgent?: string;
}

const LCB_GREEN = "rgb(148,191,115)";
const LCB_GREEN_DARK = "rgb(148,191,115)";
const LCB_GREEN_SOFT = "#EAF8EE";

const ChatSection = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "LCB Fertilizer's Query Window. Ask me any queries you have!",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isServerOnline, setIsServerOnline] = useState(true);
  const [knowledgeUrl, setKnowledgeUrl] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isUploadingKnowledge, setIsUploadingKnowledge] = useState(false);
  const [uploadedSources, setUploadedSources] = useState<KnowledgeSource[]>([]);

  // Language state: english, hinglish, or hindi
  const [language, setLanguage] = useState<"english" | "hinglish" | "hindi">("english");

  // Follow-up suggestions
  const [followUpSuggestions, setFollowUpSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isInitialLoad = useRef(true);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);
  const chipsRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  // English questions (for both english and hinglish modes)
  const englishQuestions = [
    "What is Navyakosh Organic Fertilizer?",
    "What are the benefits of using Navyakosh?",
    "How do I apply it for Wheat, Maize, and Paddy?",
    "Is it safe for long-term soil health?",
    "Can it replace chemical fertilizers?",
    "How does it improve crop yield?",
    "Where can I buy Navyakosh?",
    "How much quantity should I use?",
    "When should I apply the fertilizer?",
    "Does it work in all soil types?"
  ];

  // Hinglish questions (for hinglish mode)
  const hinglishQuestions = [
    "Navyakosh organic fertilizer kya hai?",
    "Navyakosh use karne ke kya benefits hain?",
    "Wheat, maize aur paddy ke liye kaise apply karein?",
    "Kya yeh long-term soil health ke liye safe hai?",
    "Kya yeh chemical fertilizers ko replace kar sakta hai?",
    "Yeh crop yield kaise improve karta hai?",
    "Navyakosh kahan milega?",
    "Kitni quantity use karni chahiye?",
    "Kab apply karna sahi hai?",
    "Kya yeh sabhi soil types mein kaam karta hai?"
  ];

  // Hindi questions (Devanagari script)
  const hindiQuestions = [
    "नव्याकोष जैविक उर्वरक क्या है?",
    "नव्याकोष का उपयोग करने के क्या फायदे हैं?",
    "गेहूं, मक्का और धान के लिए इसे कैसे लगाएं?",
    "क्या यह दीर्घकालिक मिट्टी के स्वास्थ्य के लिए सुरक्षित है?",
    "क्या यह रासायनिक उर्वरकों को बदल सकता है?",
    "यह फसल की पैदावार कैसे बेहतर बनाता है?",
    "नव्याकोष कहां मिलेगा?",
    "कितनी मात्रा में उपयोग करना चाहिए?",
    "कब लगाना सही है?",
    "क्या यह सभी प्रकार की मिट्टी में काम करता है?"
  ];

  const isNearBottom = (): boolean => {
    const el = scrollContainerRef.current;
    if (!el) return true;
    const threshold = 80;
    return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  };

  const scrollChatToBottom = () => {
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  useEffect(() => {
    if (isInitialLoad.current) {
      isInitialLoad.current = false;
      return;
    }
    if (isNearBottom()) scrollChatToBottom();
  }, [messages]);

  // Get current questions based on language mode
  const getCurrentQuestions = () => {
    switch (language) {
      case "english": return englishQuestions;
      case "hinglish": return [...englishQuestions, ...hinglishQuestions]; // Both English and Hinglish
      case "hindi": return hindiQuestions;
      default: return englishQuestions;
    }
  };

  // Filter suggestions based on input
  useEffect(() => {
    if (inputValue.trim().length === 0) {
      setFollowUpSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    const currentQuestions = getCurrentQuestions();
    const filtered = currentQuestions
      .filter((question) =>
        question.toLowerCase().includes(inputValue.toLowerCase())
      )
      .slice(0, 5); // Show max 5 suggestions

    setFollowUpSuggestions(filtered);
    setShowSuggestions(filtered.length > 0);
    setSelectedSuggestionIndex(-1);
  }, [inputValue, language]);

  // Keyboard navigation for suggestions
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || followUpSuggestions.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedSuggestionIndex(prev => 
        prev < followUpSuggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedSuggestionIndex(prev => prev > -1 ? prev - 1 : -1);
    } else if (e.key === "Enter" && selectedSuggestionIndex >= 0) {
      e.preventDefault();
      const selectedSuggestion = followUpSuggestions[selectedSuggestionIndex];
      setInputValue(selectedSuggestion);
      setShowSuggestions(false);
      handleSendMessage(selectedSuggestion);
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
      setSelectedSuggestionIndex(-1);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
    setShowSuggestions(false);
    handleSendMessage(suggestion);
  };

  const predefinedQuestions = getCurrentQuestions().slice(0, 10);

  // Server health check
  useEffect(() => {
    const checkServerStatus = async () => {
      const isOnline = await checkHealth();
      setIsServerOnline(isOnline);
      if (!isOnline) toast.error("AI server is currently offline.");
    };

    checkServerStatus();
    const interval = setInterval(checkServerStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const updateChipsScrollState = () => {
    const el = chipsRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 0);
    setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  };

  useEffect(() => {
    updateChipsScrollState();
    const onResize = () => updateChipsScrollState();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const scrollChips = (dir: "left" | "right") => {
    const el = chipsRef.current;
    if (!el) return;
    const amount = Math.floor(el.clientWidth * 0.9);
    el.scrollBy({ left: dir === "left" ? -amount : amount, behavior: "smooth" });
  };

  // Get language code for API
  const getLanguageForAPI = (): ApiLanguage => {
  switch (language) {
    case "english": return "en";
    case "hinglish": return "hinglish";
    case "hindi": return "hi";
    default: return "en";
  }
}

  const handleKnowledgeFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setSelectedFiles(files);
  };

  const handleUploadKnowledge = async () => {
    if (!knowledgeUrl.trim() && selectedFiles.length === 0) {
      toast.error("Add at least one URL or choose a file to upload.");
      return;
    }

    if (!isServerOnline) {
      toast.error("AI server is currently offline.");
      return;
    }

    try {
      setIsUploadingKnowledge(true);
      const urls = knowledgeUrl
        .split(/[,\n]/)
        .map((url) => url.trim())
        .filter(Boolean);

      const result = await ingestKnowledge(urls, selectedFiles);
      if (result.success) {
        toast.success(result.message || "Knowledge uploaded successfully.");
        setKnowledgeUrl("");
        setSelectedFiles([]);
        setWebsiteAssessmentResult(null);
        setUploadedSources(result.sources || []);
        if (inputRef.current) inputRef.current.focus();
      } else {
        toast.error(result.error || "Could not upload knowledge.");
        setUploadedSources([]);
      }
    } catch (error) {
      console.error(error);
      toast.error("Could not upload knowledge.");
    } finally {
      setIsUploadingKnowledge(false);
    }
  };

  const handleSendMessage = async (messageText: string) => {
    if (!messageText.trim()) return;
    if (!isServerOnline) {
      toast.error("AI server is currently offline.");
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      isUser: true,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setShowSuggestions(false);
    setIsLoading(true);
    requestAnimationFrame(scrollChatToBottom);

    try {
      const response = await sendMessage(messageText, getLanguageForAPI());
      if (response.success) {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: response.response,
          isUser: false,
          timestamp: new Date(),
          judgment: response.judgment,
          judgeExpected: true,
          activeAgent: response.active_agent,
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        toast.error(response.error || "Failed to get response.");
      }
    } catch (error) {
      console.error(error);
      toast.error("Failed to get response.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && selectedSuggestionIndex === -1) {
      e.preventDefault();
      handleSendMessage(inputValue);
    }
  };

  // Get placeholder text based on language
  const getPlaceholder = () => {
    switch (language) {
      case "english": return "Type your question...";
      case "hinglish": return "Type your question (English ya Hinglish mein)...";
      case "hindi": return "अपना प्रश्न लिखें...";
      default: return "Type your question...";
    }
  };

  // Get loading text based on language
  const getLoadingText = () => {
    switch (language) {
      case "english": return "Typing...";
      case "hinglish": return "Typing ho raha hai...";
      case "hindi": return "टाइप हो रहा है...";
      default: return "Typing...";
    }
  };

  // Get "Try asking" text based on language
  const getTryAskingText = () => {
    switch (language) {
      case "english": return "Try asking:";
      case "hinglish": return "Try asking (English/Hinglish):";
      case "hindi": return "पूछने की कोशिश करो:";
      default: return "Try asking:";
    }
  };

  // Get header subtitle based on language
  const getHeaderSubtitle = () => {
    switch (language) {
      case "english": return "Ask about Navyakosh";
      case "hinglish": return "Ask about Navyakosh (English/Hinglish)";
      case "hindi": return "नव्याकोष के बारे में पूछें";
      default: return "Ask about Navyakosh";
    }
  };

  return (
    <section className="min-h-[80vh] px-4 py-8 sm:px-6 lg:px-8 font-poppins">
      <div className="mx-auto max-w-7xl">
        <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
          <aside className="rounded-[2rem] border border-white/10 bg-slate-950/80 p-6 shadow-[0_40px_120px_-48px_rgba(16,185,129,0.65)] backdrop-blur-xl">
            <div className="space-y-4">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-emerald-300/80">Knowledge Hub</p>
                <h3 className="mt-3 text-2xl font-semibold text-white">Upload docs & website links</h3>
                <p className="mt-2 text-sm leading-6 text-slate-300">Feed your website content, PDFs, and other docs so the assistant answers from your own data.</p>
              </div>

              <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-5 space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-white">Add knowledge sources</p>
                    <p className="text-xs text-slate-400">Paste a URL or upload files.</p>
                  </div>
                  <span className={`rounded-full px-3 py-1 text-[11px] font-semibold ${isServerOnline ? 'bg-emerald-100 text-emerald-900' : 'bg-rose-100 text-rose-900'}`}>
                    {isServerOnline ? 'Server Online' : 'Server Offline'}
                  </span>
                </div>

                <div className="grid gap-3">
                  <Input
                    value={knowledgeUrl}
                    onChange={(e) => setKnowledgeUrl(e.target.value)}
                    placeholder="https://example.com"
                    className="w-full bg-slate-950/70 text-white rounded-2xl border border-white/10"
                    style={{ borderColor: 'rgba(255, 255, 255, 0.08)' }}
                    disabled={isUploadingKnowledge || !isServerOnline}
                  />
                  <label className="flex cursor-pointer items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 transition hover:border-emerald-300/60">
                    <span className="inline-flex items-center gap-2 text-emerald-200">
                      <Upload size={18} />
                      {selectedFiles.length > 0 ? `${selectedFiles.length} file(s)` : 'Choose files'}
                    </span>
                    <input type="file" multiple accept=".txt,.md,.pdf,.docx,.csv,.json,.html" className="hidden" onChange={handleKnowledgeFileChange} />
                  </label>
                </div>

                <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
                  <Button
                    onClick={handleUploadKnowledge}
                    disabled={isUploadingKnowledge || (!knowledgeUrl.trim() && selectedFiles.length === 0) || !isServerOnline}
                    className="w-full rounded-full px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-500/20"
                    style={{ backgroundColor: LCB_GREEN }}
                  >
                    {isUploadingKnowledge ? 'Uploading...' : 'Upload Knowledge'}
                  </Button>
                </div>

                {uploadedSources.length > 0 && (
                  <div className="mt-4 rounded-2xl border border-emerald-200/30 bg-emerald-50/60 p-4 text-sm text-slate-800">
                    <div className="mb-3 flex items-center justify-between gap-2">
                      <p className="font-semibold text-emerald-900">Website assessment results</p>
                      <span className="rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-semibold text-emerald-700">
                        {uploadedSources.length} source(s)
                      </span>
                    </div>
                    <div className="space-y-3">
                      {uploadedSources.map((source) => (
                        <div key={source.id} className="rounded-2xl border border-white/80 bg-white/80 p-3 shadow-sm">
                          <p className="text-sm font-semibold text-slate-900">{source.filename}</p>
                          <p className="mb-2 text-xs uppercase tracking-[0.2em] text-slate-500">{source.source_type}</p>
                          {source.assessment ? (
                            <p className="text-sm text-slate-700 whitespace-pre-line">{source.assessment}</p>
                          ) : (
                            <p className="text-sm text-slate-500">No assessment available for this source.</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-white">Explore prompts</p>
                    <p className="text-xs text-slate-400">Tap any topic to ask instantly.</p>
                  </div>
                  <div className="rounded-full bg-emerald-500/10 px-3 py-1 text-xs text-emerald-200">Smart prompts</div>
                </div>

                <div className="mt-5 grid gap-3">
                  {predefinedQuestions.map((question, index) => (
                    <button
                      key={index}
                      onClick={() => handleSendMessage(question)}
                      disabled={isLoading || !isServerOnline}
                      className="w-full rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-left text-sm text-slate-100 transition hover:border-emerald-300/60 hover:bg-slate-900"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </aside>

          <div className="rounded-[2rem] border border-white/10 bg-white/95 shadow-2xl shadow-slate-950/10 overflow-hidden flex h-full flex-col">
            <div className="flex flex-col gap-3 border-b border-slate-200/70 bg-gradient-to-r from-emerald-600 to-emerald-500 px-6 py-5 text-white">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-emerald-100/90">General Chat · Supervisor enabled</p>
                  <h2 className="text-2xl font-semibold">Ask anything—your request will reach the right agent</h2>
                </div>
                <div className="rounded-full bg-white/10 px-4 py-2 text-sm text-white shadow-inner shadow-black/10">
                  {getHeaderSubtitle()}
                </div>
              </div>
              <p className="max-w-2xl text-sm text-emerald-100/90">The supervisor automatically routes marketing, tracker, product, and document questions to the appropriate specialist.</p>
            </div>

            <div ref={scrollContainerRef} className="min-h-[420px] flex-1 overflow-y-auto bg-slate-50 p-6">
              <div className="space-y-4">
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="rounded-3xl bg-emerald-50 px-5 py-3 text-sm font-medium text-emerald-900 shadow-sm">
                      {getLoadingText()}
                    </div>
                  </div>
                )}
              </div>
              <div ref={messagesEndRef} />
            </div>

            <div className="space-y-4 bg-white/95 border-t border-slate-200/80 px-6 py-5">
              {showSuggestions && followUpSuggestions.length > 0 && (
                <div className="rounded-3xl border border-slate-200/70 bg-slate-100 p-3 shadow-sm">
                  <div className="grid gap-1">
                    {followUpSuggestions.map((suggestion, index) => (
                      <button
                        key={index}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className={`w-full rounded-2xl px-4 py-3 text-left text-sm text-slate-700 transition ${selectedSuggestionIndex === index ? 'bg-emerald-50 text-emerald-900' : 'hover:bg-slate-100'}`}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex flex-col gap-3 sm:flex-row">
                <Input
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  onKeyDown={handleKeyDown}
                  placeholder={getPlaceholder()}
                  className="flex-1 rounded-3xl border border-slate-200 bg-white px-4 py-3 text-slate-900 shadow-sm"
                  style={{ borderColor: 'rgba(148,191,115,0.18)' }}
                  disabled={isLoading || !isServerOnline}
                  autoComplete="off"
                />
                <Button
                  onClick={() => handleSendMessage(inputValue)}
                  disabled={isLoading || !inputValue.trim() || !isServerOnline}
                  className="inline-flex items-center justify-center rounded-full bg-emerald-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-500/20 transition hover:shadow-emerald-500/30 disabled:cursor-not-allowed disabled:bg-emerald-300"
                >
                  <ArrowRight size={18} />
                </Button>
              </div>

              <div className="relative">
                <div className="absolute left-0 top-1/2 hidden -translate-y-1/2 sm:block">
                  <button
                    aria-label="Scroll left"
                    onClick={() => scrollChips('left')}
                    disabled={!canScrollLeft}
                    className={`rounded-full p-2 shadow-lg transition ${canScrollLeft ? 'bg-white text-slate-700 hover:scale-105' : 'bg-slate-100 text-slate-400 cursor-not-allowed'}`}
                  >
                    <ChevronLeft size={16} />
                  </button>
                </div>

                <div
                  ref={chipsRef}
                  onScroll={updateChipsScrollState}
                  className="flex gap-2 overflow-x-auto rounded-3xl border border-slate-200 bg-slate-50 px-3 py-3 scrollbar-hide"
                  style={{ WebkitOverflowScrolling: 'touch' }}
                >
                  {predefinedQuestions.map((question, index) => (
                    <button
                      key={index}
                      onClick={() => handleSendMessage(question)}
                      disabled={isLoading || !isServerOnline}
                      className="shrink-0 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50"
                    >
                      {question}
                    </button>
                  ))}
                </div>

                <div className="absolute right-0 top-1/2 hidden -translate-y-1/2 sm:block">
                  <button
                    aria-label="Scroll right"
                    onClick={() => scrollChips('right')}
                    disabled={!canScrollRight}
                    className={`rounded-full p-2 shadow-lg transition ${canScrollRight ? 'bg-white text-slate-700 hover:scale-105' : 'bg-slate-100 text-slate-400 cursor-not-allowed'}`}
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ChatSection;
