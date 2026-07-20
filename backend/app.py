from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import uuid
try:
    import openai
    OPENAI_IMPORTED = True
    OPENAI_IMPORT_ERROR = None
except Exception as e:
    openai = None
    OPENAI_IMPORTED = False
    OPENAI_IMPORT_ERROR = str(e)

try:
    from rag_system import RAGSystem
except Exception as e:
    RAGSystem = None
    RAG_IMPORT_ERROR = str(e)
else:
    RAG_IMPORT_ERROR = None

# Load environment variables from the backend folder (and optionally the workspace root)
BASE_DIR = Path(__file__).resolve().parent
for env_path in [BASE_DIR / '.env', BASE_DIR.parent / '.env']:
    if env_path.exists():
        load_dotenv(env_path, override=False)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'chat_logs.log'),
        logging.StreamHandler()
    ]
)

# Logs directory
logs_dir = BASE_DIR / 'logs'
logs_dir.mkdir(exist_ok=True)

def log_message(user_id, message, is_user=True, response=None, error=None):
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'user_id': user_id,
        'message_type': 'user' if is_user else 'ai',
        'message': message,
        'response': response,
        'error': error,
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown')
    }
    log_file = logs_dir / f'chat_logs_{datetime.now().strftime("%Y-%m-%d")}.json'
    try:
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to write to log file: {e}")

    if is_user:
        logging.info(f"User {user_id} ({request.remote_addr}): {message}")
    else:
        # Response may be None when an error occurs; guard against slicing None
        if response:
            snippet = response[:100] + ('...' if len(response) > 100 else '')
        else:
            snippet = '<no response>'
        logging.info(f"AI Response to {user_id}: {snippet}")

def get_user_id():
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


def build_fallback_answer(message: str, relevant_context: str, personal_info: dict) -> str:
    text = (message or '').lower()
    name = personal_info.get('name', 'This brand')

    if any(word in text for word in ['buy', 'purchase', 'order', 'where can']):
        return (
            f"You can buy {name} through authorized dealers, agri-stores, or online at "
            "Amazon: https://amzn.in/d/hBRlaGo. For bulk orders, contact +91 91988 03978."
        )

    if any(word in text for word in ['crop', 'crops', 'which crop']):
        return "It works well for wheat, maize, rice, pulses, cotton, sugarcane, fruits, vegetables, and spices."

    if any(word in text for word in ['safe', 'soil health', 'chemical', 'organic']):
        return "Yes, it is 100% organic and safe for long-term soil health. It reduces dependence on chemical fertilizers and supports sustainable farming."

    if any(word in text for word in ['water', 'irrigation', 'retain']):
        return "It improves water retention and can reduce irrigation needs by up to 33%."

    if any(word in text for word in ['how does', 'improve', 'benefit', 'microbe', 'mechanism']):
        return "It restores microbial balance in the soil, improves fertility, and helps plants absorb nutrients more effectively."

    if any(word in text for word in ['result', 'yield', 'quality']):
        return "Farmers often report higher yields, healthier crops, and better soil quality. The product is designed to improve productivity sustainably."

    if any(word in text for word in ['what is', 'about', 'brand', 'who']):
        return (
            f"{name} is a smart organic fertilizer solution for sustainable farming. "
            "It improves soil health, supports better crop growth, and is safe for crops, farmers, and the environment."
        )

    return (
        f"{name} is a smart organic fertilizer solution focused on sustainable farming. "
        "It improves soil fertility, water retention, and crop performance while reducing reliance on chemical inputs."
    )

# Configure Gemini
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key
    genai.configure(api_key=api_key)
# Allow overriding the Gemini model via env, provide a safe default
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

# Configure OpenAI fallback (optional)
openai_api_key = os.getenv('OPENAI_API_KEY')
if OPENAI_IMPORTED and openai_api_key:
    openai.api_key = openai_api_key
elif openai_api_key and not OPENAI_IMPORTED:
    print(f"⚠️ OPENAI_API_KEY configured but openai library is missing: {OPENAI_IMPORT_ERROR}")

# Helper to call OpenAI as a fallback
def call_openai(prompt: str) -> str:
    if not OPENAI_IMPORTED or not openai_api_key:
        raise RuntimeError('OpenAI fallback not available')
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise FAQ assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise

# Initialize RAG
rag_system = None
if api_key and RAGSystem is not None:
    try:
        rag_system = RAGSystem(api_key)
        rag_system.build_vectorstore()
    except Exception as e:
        print(f"⚠️ RAG initialization failed: {e}")
        rag_system = None
elif RAG_IMPORT_ERROR:
    print(f"⚠️ RAG module unavailable: {RAG_IMPORT_ERROR}")

print("🔍 Environment check:")
print(f"   GEMINI_API_KEY from env: {'✅ Found' if api_key else '❌ Not found'}")
print(f"   GEMINI_MODEL: {GEMINI_MODEL}")
print()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        user_id = get_user_id()
        log_message(user_id, message, is_user=True)

        if not api_key:
            error_msg = 'Gemini API key not configured.'
            log_message(user_id, message, is_user=False, error=error_msg)
            return jsonify({'error': error_msg, 'success': False}), 500

        # If RAG isn't available, continue in a degraded mode: use Gemini directly
        if rag_system:
            personal_info = rag_system.get_personal_info()
            profile_summary = rag_system.get_summary_document()
        else:
            logging.warning('RAG not initialized — serving responses without retrieval augmentation.')
            personal_info = {'name': 'Brand', 'title': 'Brand Assistant'}
            profile_summary = ''

        # Refine Query
        query_refiner_prompt = f"""
You are a research assistant. Refine the user question into a precise search query for the brand knowledge base.

User's Original Question: "{message}"
Refined Search Query:
"""
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            query_refiner_response = model.generate_content(query_refiner_prompt)
            refined_query = getattr(query_refiner_response, 'text', str(query_refiner_response)).strip()
            print(f"🧠 Refined Search Query: {refined_query}")
        except Exception as e:
            print(f"⚠️ Query refinement failed (model={GEMINI_MODEL}): {e}")
            refined_query = message

        # Search RAG
        if rag_system:
            try:
                relevant_context = rag_system.search_relevant_context(refined_query, k=4)
                print("Retrieved relevant context: ", relevant_context)
            except Exception as e:
                print(f"⚠️ RAG search failed: {e}")
                relevant_context = "Unable to retrieve relevant information."
        else:
            relevant_context = ""

        # Final Answer (short + direct)
        final_answer_prompt = f"""
You are a precise FAQ assistant for the brand {personal_info['name']}.

<USER_QUESTION>
{message}
</USER_QUESTION>

<DETAILED_CONTEXT>
{relevant_context}
</DETAILED_CONTEXT>

INSTRUCTIONS:
- Answer in **2–5 sentences maximum**.
- If the context has a clearly written "Ans.", return it verbatim.
- Do NOT add much extra explanations.
- If no relevant answer exists, say: "Sorry, I don’t have that information right now."
"""

        ai_response = None
        # First try Gemini if available
        if api_key:
            try:
                final_model = genai.GenerativeModel(GEMINI_MODEL)
                final_response = final_model.generate_content(final_answer_prompt)
                ai_response = getattr(final_response, 'text', None)
                if ai_response is None:
                    ai_response = str(final_response)
                ai_response = ai_response.strip()
            except Exception as e:
                print(f"⚠️ Gemini final model call failed (model={GEMINI_MODEL}): {e}")
                ai_response = None

        # Fallback to OpenAI if Gemini failed
        if not ai_response and OPENAI_IMPORTED and openai_api_key:
            try:
                ai_response = call_openai(final_answer_prompt)
            except Exception as e:
                print(f"⚠️ OpenAI fallback failed: {e}")
                ai_response = None

        # Final local fallback: answer from the loaded knowledge base
        if not ai_response:
            ai_response = build_fallback_answer(message, relevant_context, personal_info)

        log_message(user_id, message, is_user=False, response=ai_response)

        return jsonify({
            'response': ai_response,
            'success': True,
            'refined_query': refined_query,
            'session_id': user_id
        })

    except Exception as e:
        error_msg = f'Failed to get AI response: {str(e)}'
        user_id = get_user_id()
        log_message(user_id, message if 'message' in locals() else 'Unknown', is_user=False, error=error_msg)
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to get AI response', 'success': False}), 500

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    api_key_status = "configured" if api_key else "not configured"
    rag_status = "initialized" if rag_system else "not initialized"
    return jsonify({'status': 'healthy','api_key': api_key_status,'rag_system': rag_status})

# Rebuild vectorstore
@app.route('/api/rebuild-vectorstore', methods=['POST'])
def rebuild_vectorstore():
    try:
        if not api_key:
            return jsonify({'error': 'Gemini API key not configured','success': False}), 500
        global rag_system
        rag_system = RAGSystem(api_key)
        rag_system.build_vectorstore()
        return jsonify({'message': 'Vector database rebuilt','success': True})
    except Exception as e:
        print(f"Error rebuilding vectorstore: {str(e)}")
        return jsonify({'error': 'Failed to rebuild vector database','success': False}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    print("🚀 Starting AI Assistant Backend with RAG (Gemini)...")
    print(f"📡 API Key Status: {'✅ Configured' if api_key else '❌ Not configured'}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")
    print(f"🌐 Server running at: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
