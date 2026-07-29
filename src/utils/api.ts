// Backend API configuration - defaults to the local backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001';

export interface ChatResponse {
  response: string;
  success: boolean;
  error?: string;
}

export interface IngestResponse {
  success: boolean;
  message?: string;
  count?: number;
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
