// Backend API configuration - defaults to the local backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001';

export interface ChatJudgment {
    status: "completed" | "fallback" | "unavailable" | "invalid_output" | "disabled";
    verdict: "pass" | "warning" | "fail" | "not_judged";
    score: number | null;
    dimensions?: Record<string, number>;
    issues?: string[];
    missing_information?: string[];
    feedback?: string;
    agent?: string;
    error?: string;
    evaluation_method?: "llm" | "deterministic_fallback";
}

export interface ChatResponse {
  response: string;
  success: boolean;
  judgment?: ChatJudgment;
  active_agent?: string;
  routing?: {
    agent: "general" | "marketing" | "competitive_intelligence" | "tracker";
    display_name: string;
    reason: string;
    confidence: number;
    routing_method: "llm" | "fallback";
  };
  error?: string;
}

export interface KnowledgeSource {
  id: string;
  filename: string;
  source_type: string;
  assessment?: string;
}

export interface IngestResponse {
  success: boolean;
  message?: string;
  count?: number;
  sources?: KnowledgeSource[];
  error?: string;
}

export type ApiLanguage = "en" | "hi" | "hinglish";

export interface AuthResponse {
  success: boolean;
  message?: string;
  token?: string;
  user?: {
    name: string;
    role: "admin" | "user";
    email?: string;
  };
  error?: string;
}

export type PlanStatus = "not_started" | "in_progress" | "complete";
export interface MarketingPlan {
  id: number;
  title: string;
  strategy: string;
  owner: string;
  status: PlanStatus;
  created_by: string;
  created_at: string;
  updated_at: string;
}

const SESSION_STORAGE_KEY = "lcb_session_id";

function getSessionId(): string | null {
  return localStorage.getItem(SESSION_STORAGE_KEY);
}

function setSessionId(sessionId: string) {
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

function getAuthHeaders(extraHeaders: Record<string, string> = {}, includeJsonContentType = true) {
  const token = localStorage.getItem("lcb_auth_token");
  const sessionId = getSessionId();
  return {
    ...(includeJsonContentType ? { "Content-Type": "application/json" } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(sessionId ? { "X-Session-ID": sessionId } : {}),
    ...extraHeaders,
  };
}

export async function sendMessage(
  message: string,
  language: ApiLanguage = "en"
): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        ...getAuthHeaders(),
        "X-Language": language,
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      if (response.status === 401) {
        // The user may have an expired token after the eight-hour session window.
        localStorage.removeItem("lcb_auth_token");
        localStorage.removeItem("lcb_user");
      }
      return {
        response: "",
        success: false,
        error: response.status === 401
          ? "Your session has expired. Please log in again."
          : (errorData?.error || `Request failed (HTTP ${response.status})`),
      };
    }

    const data = await response.json();
    if (data?.session_id) {
      setSessionId(data.session_id);
    }
    return data;
  } catch (error) {
    console.error("Error sending message:", error);
    return {
      response:
        "Sorry, I'm having trouble connecting to the server. Please make sure the backend is running on http://localhost:5001",
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

export async function ingestKnowledge(urls: string[], files: File[]): Promise<IngestResponse> {
  try {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    urls.filter(Boolean).forEach((url) => formData.append("urls", url));

    const response = await fetch(`${API_BASE_URL}/api/ingest`, {
      method: "POST",
      headers: getAuthHeaders({}, false),
      body: formData,
    });

    const data = await response.json().catch(() => null);
    if (!response.ok) {
      return {
        success: false,
        error: data?.error || `HTTP error! status: ${response.status}`,
      };
    }

    if (data?.session_id) {
      setSessionId(data.session_id);
    }

    return data || { success: false, error: "Empty response from ingestion API" };
  } catch (error) {
    console.error("Knowledge ingestion failed:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      return false;
    }
    const data = await response.json();
    return data.status === "healthy";
  } catch (error) {
    console.error("Health check failed:", error);
    return false;
  }
}

export async function assessWebsite(url: string): Promise<{ success: boolean; url?: string; assessment?: string; error?: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/assess-website`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ url }),
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      return {
        success: false,
        error: data?.error || `Request failed (HTTP ${response.status})`,
      };
    }
    return data || { success: false, error: "Empty response from our assessment API" };
  } catch (error) {
    console.error("Website assessment failed:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

export interface AgentPrompt {
  id: number;
  agent_key: string;
  display_name: string;
  prompt_text: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentListItem {
  id: number;
  filename: string;
  source_type: string;
  created_at: string;
  content_preview?: string;
}

export interface VectorStoreInfo {
  status: {
    collection_name: string;
    retrieval_mode: string;
    vectorstore_ready: boolean;
    embedding_model: string;
    embeddings_initialized: boolean;
    storage_path: string;
    persisted_vector_count: number | null;
  };
  statistics: {
    total_chunks: number;
    filtered_chunks: number;
    total_characters: number;
    average_chunk_characters: number;
    source_counts: Record<string, number>;
  };
  pagination: { page: number; page_size: number; total_pages: number };
  chunks: Array<{
    id: number;
    chunk_index: number | string;
    source_type: string;
    source_label: string;
    character_count: number;
    word_count: number;
    content: string;
    metadata: Record<string, unknown>;
  }>;
}

export async function getAgentPrompts(): Promise<AgentPrompt[]> {
  const response = await fetch(`${API_BASE_URL}/api/agent-prompts`, {
    method: "GET",
    headers: getAuthHeaders(),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || `Failed to load agent prompts (HTTP ${response.status})`);
  }
  const data = await response.json();
  return data.prompts || [];
}

export async function saveAgentPrompt(agent_key: string, display_name: string, prompt_text: string): Promise<AgentPrompt> {
  const response = await fetch(`${API_BASE_URL}/api/agent-prompts/${agent_key}`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({ display_name, prompt_text }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || `Failed to save prompt (HTTP ${response.status})`);
  }
  const data = await response.json();
  return data.prompt;
}

export async function getDocuments(): Promise<DocumentListItem[]> {
  const response = await fetch(`${API_BASE_URL}/api/documents`, {
    method: "GET",
    headers: getAuthHeaders(),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || `Failed to load documents (HTTP ${response.status})`);
  }
  const data = await response.json();
  return data.documents || [];
}

export async function getVectorStoreInfo(page = 1, search = ""): Promise<VectorStoreInfo> {
  const params = new URLSearchParams({ page: String(page), page_size: "25" });
  if (search.trim()) params.set("search", search.trim());
  const response = await fetch(`${API_BASE_URL}/api/vectorstore-info?${params}`, {
    method: "GET",
    headers: getAuthHeaders(),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.error || `Failed to inspect vector database (HTTP ${response.status})`);
  return data as VectorStoreInfo;
}

export async function deleteDocument(docId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${docId}`, {
    method: "DELETE",
    headers: getAuthHeaders(),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || `Failed to delete document (HTTP ${response.status})`);
  }
}

export async function rebuildVectorstore(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/rebuild-vectorstore`, {
    method: "POST",
    headers: getAuthHeaders(),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || `Failed to rebuild vectorstore (HTTP ${response.status})`);
  }
}

async function marketingRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { ...getAuthHeaders(), ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.success) throw new Error(data.error || "Request failed");
  return data as T;
}

export async function sendMarketingMessage(message: string): Promise<ChatResponse> {
  return marketingRequest<ChatResponse>("/api/marketing/chat", {
    method: "POST", body: JSON.stringify({ message }),
  });
}

export async function sendCompetitorMessage(message: string): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 80000);
  try {
    return await marketingRequest<ChatResponse>("/api/competitors/chat", {
      method: "POST", body: JSON.stringify({ message }), signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("The local AI model took too long to respond. Please try a more specific competitor question.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function askTrackerQuestion(question: string): Promise<ChatResponse> {
  return marketingRequest<ChatResponse>("/api/tracker/chat", {
    method: "POST", body: JSON.stringify({ question }),
  });
}

export async function saveMarketingPlan(title: string, strategy: string, owner = ""): Promise<MarketingPlan> {
  const data = await marketingRequest<{ plan: MarketingPlan }>("/api/marketing/plans", {
    method: "POST", body: JSON.stringify({ title, strategy, owner }),
  });
  return data.plan;
}

export async function getMarketingPlans(): Promise<MarketingPlan[]> {
  const data = await marketingRequest<{ plans: MarketingPlan[] }>("/api/marketing/plans");
  return data.plans;
}

export async function updateMarketingPlan(id: number, status: PlanStatus): Promise<MarketingPlan> {
  const data = await marketingRequest<{ plan: MarketingPlan }>(`/api/marketing/plans/${id}`, {
    method: "PATCH", body: JSON.stringify({ status }),
  });
  return data.plan;
}
