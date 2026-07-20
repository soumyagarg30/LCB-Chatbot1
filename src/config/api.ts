// API configuration file - prepared for backend separation
// When you implement the Python backend, change BASE_URL to your backend server address

// Prefer Vite override when available, otherwise fall back to the local backend
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL) || (process.env.NODE_ENV === 'production'
  ? 'https://lcb-backend-tjgz.onrender.com'
  : 'http://localhost:5001');

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  ENDPOINTS: {
    CHAT: '/api/chat',
    HEALTH: '/api/health',
    CONTACT: '/api/contact'
  }
};

// API call function prepared for Python backend
export async function callPythonBackend(message: string): Promise<string> {
  try {
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CHAT}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
        user_id: 'anonymous', // optional: replace with logged-in user id if available
        timestamp: new Date().toISOString()
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();
    return data.response || data.message || "Sorry, I couldn't process your request.";
  } catch (error) {
    console.error('Error calling Python backend:', error);
    throw error;
  }
}
