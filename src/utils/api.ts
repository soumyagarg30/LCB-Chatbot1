// Backend API configuration - defaults to the local backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001';

export interface ChatResponse {
  response: string;
  success: boolean;
  error?: string;
}

export async function sendMessage(
  message: string,
  language: "en" | "hi" = "en"  
): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Language": language, 
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
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
