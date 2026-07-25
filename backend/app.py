from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import traceback
import requests
from pathlib import Path
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from datetime import datetime
import uuid
from auth_utils import create_token, verify_token, get_secret
from db_utils import init_db, create_user, verify_user, store_document, get_documents_content
from knowledge_ingester import _read_local_file
from llm_client import LLMClient

# Load environment variables from the backend folder (and optionally the workspace root)
BASE_DIR = Path(__file__).resolve().parent
for env_path in [BASE_DIR / '.env', BASE_DIR.parent / '.env']:
    if env_path.exists():
        load_dotenv(env_path, override=False)

try:
    from rag_system import RAGSystem
except Exception as e:
    RAGSystem = None
    RAG_IMPORT_ERROR = str(e)
else:
    RAG_IMPORT_ERROR = None

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

# Uploads directory for document ingestion
uploads_dir = BASE_DIR / 'uploads'
uploads_dir.mkdir(exist_ok=True)

# Initialize database
init_db()

SESSION_HISTORY = {}
MAX_SESSION_HISTORY = 8


def get_session_history(user_id):
    return SESSION_HISTORY.get(user_id, [])


def append_session_history(user_id, role, text):
    if not text:
        return
    history = SESSION_HISTORY.setdefault(user_id, [])
    history.append({'role': role, 'text': text.strip()})
    if len(history) > MAX_SESSION_HISTORY:
        del history[:-MAX_SESSION_HISTORY]


def get_previous_user_questions(user_id, max_items=4):
    history = get_session_history(user_id)
    questions = [entry['text'] for entry in history if entry['role'] == 'user']
    return questions[-max_items:]


def get_conversation_history(user_id, max_messages=8):
    history = get_session_history(user_id)
    if not history:
        return ""

    trimmed = history[-max_messages:]
    rendered = []
    for entry in trimmed:
        role_label = 'User' if entry['role'] == 'user' else 'Assistant'
        rendered.append(f"{role_label}: {entry['text']}")
    return "\n".join(rendered)


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


def get_bearer_token():
    header = request.headers.get('Authorization', '')
    if header.startswith('Bearer '):
        return header.split(' ', 1)[1].strip()
    return None


def require_auth(required_role=None):
    token = get_bearer_token()
    if not token:
        return None

    payload = verify_token(token, get_secret())
    if not payload:
        return None

    if required_role and payload.get('role') != required_role:
        return None

    return payload


def clean_llm_output(text: str) -> str:
    """
    Safety net: strip any reasoning/thinking traces that a model might still
    emit inline (e.g. <think>...</think> blocks, or stray leading braces),
    so the chat widget only ever shows the final answer.
    """
    if not text:
        return text
    # Remove <think>...</think> blocks (some models wrap reasoning like this)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove a stray leading "thinking" label/object if it ever leaks through
    text = re.sub(r'^\s*thinking\s*\{.*?\}\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def clean_prose_output(text: str) -> str:
    """Normalize model output to remove headings, preserve line breaks, and keep full prose."""
    if not text:
        return text
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'(?m)^(Answer|Explanation|Evidence|Next steps)\s*[:\-]\s*', '', text)
    text = re.sub(r'\s*•\s*', '\n• ', text)
    text = re.sub(r'\s*\*\s*', '\n* ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def sanitize_retrieved_context(text: str) -> str:
    """Sanitize retrieved context before sending to LLM or local synthesizer.

    Removes large JSON-like blobs, inline site navigation JSON, and HTML tags
    that often leak into retrieved snippets. Keeps readable sentences and
    short quoted snippets like [Source: filename].
    """
    if not text:
        return text
    # Remove script and style blocks first to avoid leaving JS as plain text
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove JSON key-value pairs like "key":"value" (common in dumped nav)
    text = re.sub(r'"[A-Za-z0-9_\-]{1,60}"\s*:\s*"[^"]{0,300}"', ' ', text)
    # Remove small JSON objects or arrays that look like navigation dumps
    text = re.sub(r'\{[^\}]{0,800}\}', ' ', text)
    text = re.sub(r'\[[^\]]{0,800}\]', ' ', text)
    # Collapse repeated punctuation or long non-word runs
    text = re.sub(r'[^\w\s\[\]\.:,\-\%\(\)\/]{3,}', ' ', text)

    # Break into lines and keep only lines that look like natural language (filter out code/navigation dumps)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    good_lines = []
    for l in lines:
        # remove question/answer formatting if present
        l = re.sub(r'^\s*(Q|Question)\s*[:\-]\s*', '', l, flags=re.IGNORECASE)
        l = re.sub(r'^\s*(A|Answer)\s*[:\-]\s*', '', l, flags=re.IGNORECASE)
        # skip lines that look like code or JSON keys
        if re.search(r'\b(function|window|document|var|let|const|if|else|return|console|=>|require|module|exports)\b', l, re.IGNORECASE):
            continue
        if re.search(r'"[A-Za-z0-9_\-]{1,40}"\s*:', l):
            continue
        # require at least 3 words and some letters
        if len(l.split()) < 3:
            continue
        if len(re.findall(r'[a-zA-Z]', l)) < 5:
            continue
        good_lines.append(l)

    if good_lines:
        text = '\n'.join(good_lines)
    else:
        # fallback to a short placeholder when no good textual lines found
        text = 'Website content (HTML/JS removed) — visit the source or re-upload a plain text/PDF for better extraction.'

    # Convert inline bullets to multiline bullet lists
    text = re.sub(r'\s*•\s*', '\n• ', text)
    text = re.sub(r'\s*–\s*', '\n- ', text)
    text = re.sub(r'\s*\*\s*', '\n* ', text)

    # Normalize whitespace while preserving line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    # Shorten overly long sanitized text
    if len(text) > 4000:
        text = text[:4000].rsplit(' ', 1)[0] + '...'
    return text


def build_fallback_answer(message: str, relevant_context: str, personal_info: dict) -> str:
    """Used only when there is genuinely no usable context. Keeps the user informed
    instead of guessing, since guessing without context previously caused bugs
    (undefined variables `name`/`text` in dead code below the old return)."""
    name = personal_info.get('name', 'this brand')
    context = (relevant_context or '').strip()
    if not context or context.startswith('No additional context available'):
        return (
            "I don't have enough information from the uploaded documents to answer that accurately. "
            "Please clarify your question or provide the relevant knowledge so I can answer using the provided sources."
        )

    return (
        f"I found relevant information in the uploaded documents, but I'm not fully confident it answers your "
        f"question about {name}. Could you clarify whether you're asking about benefits, usage, crop suitability, "
        f"or another specific aspect?"
    )


def should_ask_follow_up(message: str, relevant_context: str, brand_name: str) -> bool:
    """Return True when the user's question is broad and the retrieved context is weak."""
    text = (message or "").strip().lower()
    context = (relevant_context or "").strip()
    if not context:
        return True

    broad_markers = ["what is", "tell me about", "explain", "overview", "summary", "how does it work", "benefits", "details"]
    specific_markers = ["dosage", "recommendation", "for maize", "for wheat", "for rice", "price", "cost", "where to buy", "buy"]

    if any(marker in text for marker in specific_markers):
        return False
    if any(marker in text for marker in broad_markers):
        return len(context.split()) < 80
    return False


def build_contextual_answer_prompt(
    message: str,
    relevant_context: str,
    personal_info: dict,
    previous_questions: list[str] | None = None,
    conversation_history: str | None = None,
    use_history_only: bool = False,
) -> str:
    """Build a reusable prompt that encourages deeper, context-aware answers."""
    brand_name = personal_info.get('name', 'this brand')
    context = (relevant_context or '').strip()
    if not context:
        context = "No prior conversation history is available."

    guidance = "If the user asks a broad or ambiguous question, ask one short clarifying question before giving a final answer."
    if not should_ask_follow_up(message, context, brand_name):
        guidance = "Use the available context directly and give a confident, detailed answer."

    context_label = "RETRIEVED_KNOWLEDGE"
    if use_history_only:
        guidance = "Use only the past conversation history to answer. Do not use any knowledge base documents or external sources."
        context_label = "PAST CONVERSATION HISTORY"

    previous_questions_section = ""
    if conversation_history:
        previous_questions_section = "PAST CONVERSATION HISTORY:\n"
        previous_questions_section += conversation_history.strip() + "\n\n"
    elif previous_questions:
        previous_questions_section = "PAST USER QUESTIONS:\n"
        previous_questions_section += "\n".join(f"- {q}" for q in previous_questions)
        previous_questions_section += "\n\n"

    prompt_path = Path(__file__).resolve().parent / 'prompts' / 'contextual_answer_prompt.txt'
    template = prompt_path.read_text(encoding='utf-8')
    return template.format(
        brand_name=brand_name,
        user_question=message,
        retrieved_context=context,
        previous_questions_section=previous_questions_section,
        clarification_guidance=guidance,
        context_label=context_label,
    )


def synthesize_local_answer(message: str, relevant_context: str, personal_info: dict) -> str:
    """Create a structured, context-aware answer from retrieved snippets when LLMs are unavailable.

    Heuristic synthesizer: extracts source tags, selects up to 3 informative snippets,
    and returns formatted prose. This is a FALLBACK ONLY — if you're consistently seeing
    this output, it means the DeepSeek call is failing (check server logs for
    "Deepseek call failed").
    """
    ctx = sanitize_retrieved_context((relevant_context or "").strip())
    name = personal_info.get('name', 'this brand')

    if not ctx or ctx.startswith('No additional context available'):
        return build_fallback_answer(message, relevant_context, personal_info)

    sources = re.findall(r"\[Source:\s*([^\]]+)\]", ctx)
    dedup_sources = []
    for s in sources:
        if s not in dedup_sources:
            dedup_sources.append(s)

    cleaned = re.sub(r"\[Source:[^\]]+\]", "", ctx)
    cleaned = re.sub(r"Relevant Information \d+:", "", cleaned)
    cleaned = re.sub(r"^\s*(Q|Question)\s*[:\-].*?A\s*[:\-]", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    candidates = [c.strip() for c in re.split(r"\n\n+", cleaned) if c.strip()]
    if not candidates:
        candidates = [cleaned[:1000]] if cleaned else []

    picked = []
    for c in candidates:
        if len(c) < 30:
            continue
        if not re.search(r"[a-zA-Z]{2,}", c):
            continue
        if c in picked:
            continue
        picked.append(c)
        if len(picked) >= 3:
            break

    primary = picked[0] if picked else (cleaned[:500] if cleaned else "")
    sentences = re.split(r'(?<=[.!?])\s+', primary)
    answer_sent = ""
    for s in sentences:
        if len(s) > 20 and re.search(r"\b(is|are|was|were|lead|leads|show|shows|demonstrat|increase|improve|reduce|result|enhanc|yield|contains|provides)\b", s, re.IGNORECASE):
            answer_sent = s.strip()
            break
    if not answer_sent:
        answer_sent = sentences[0].strip() if sentences else (primary[:200] + '...')

    normalized_message = re.sub(r'\s+', ' ', (message or '').strip().lower())
    normalized_answer = re.sub(r'\s+', ' ', answer_sent.strip().lower())
    if normalized_message and (normalized_answer.startswith(normalized_message) or answer_sent.strip().endswith('?')):
        return build_fallback_answer(message, relevant_context, personal_info)

    detail_parts = []
    for p in picked[1:4]:
        short = p.strip()
        if len(short) > 280:
            short = short[:277].rsplit(' ', 1)[0] + '...'
        if short and short not in detail_parts:
            detail_parts.append(short)

    evidence_note = "This response is drawn from the stored documents and uploaded knowledge base material."

    lines = [answer_sent]
    for detail in detail_parts:
        lines.append(detail)
    if dedup_sources:
        lines.append(f"Sources: {', '.join(dedup_sources[:3])}.")
    lines.append(evidence_note)

    return clean_prose_output("\n\n".join(lines))


# Default is local Ollama with llama3.2:3b. See env.example for other providers.
llm_client = LLMClient()
LLM_PROVIDER = llm_client.config.provider


# Initialize RAG lazily so backend can still start even if RAG build fails on startup
rag_system = None
if RAG_IMPORT_ERROR:
    print(f"⚠️ RAG module unavailable: {RAG_IMPORT_ERROR}")


def get_rag_system():
    global rag_system
    if rag_system is not None:
        return rag_system
    if RAGSystem is None:
        return None
    try:
        rag_system = RAGSystem()
        rag_system.build_vectorstore()
        return rag_system
    except Exception as e:
        print(f"⚠️ RAG initialization failed: {e}")
        traceback.print_exc()  # print full stack trace so the real cause is visible in logs
        rag_system = None
        return None


# Initialize RAG early so the backend reports its availability correctly.
# This initialization is safe in offline mode because RAGSystem falls back to keyword search when embeddings are unavailable.
get_rag_system()

print("🔍 Environment check:")
print(f"   LLM_PROVIDER: {LLM_PROVIDER}")
print(f"   LLM_MODEL: {llm_client.config.model}")
print(f"   LLM_BASE_URL: {llm_client.config.base_url}")
print(f"   RAG_SYSTEM: {'✅ Ready' if rag_system else '❌ Not ready'}")
print()


@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', '').strip()

    if not email or not password or not name:
        return jsonify({'error': 'Email, password, and name are required', 'success': False}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters', 'success': False}), 400

    user = create_user(email, password, name)
    if not user:
        return jsonify({'error': 'Email already registered', 'success': False}), 409

    token = create_token({'sub': user['email'], 'role': user['role'], 'name': user['name']}, get_secret())
    return jsonify({
        'success': True,
        'message': 'Account created successfully',
        'token': token,
        'user': user
    }), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required', 'success': False}), 400

    user = verify_user(email, password)
    if not user:
        return jsonify({'error': 'Invalid email or password', 'success': False}), 401

    token = create_token({'sub': user['email'], 'role': user['role'], 'name': user['name']}, get_secret())
    return jsonify({
        'success': True,
        'message': 'Signed in successfully',
        'token': token,
        'user': user
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    message = None
    try:
        auth_user = require_auth()
        if not auth_user:
            return jsonify({'error': 'Authentication required', 'success': False}), 401
        data = request.get_json()
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        user_id = get_user_id()
        append_session_history(user_id, 'user', message)
        log_message(user_id, message, is_user=True)

        rag = get_rag_system()
        if rag:
            personal_info = rag.get_personal_info()
        else:
            logging.warning('RAG not initialized — serving responses without retrieval augmentation.')
            personal_info = {'name': 'Brand', 'title': 'Brand Assistant'}

        conversation_history = get_conversation_history(user_id, max_messages=8)
        # Always retrieve seeded FAQs and uploaded-document chunks. Conversation history
        # adds continuity; it must not replace the actual knowledge base.
        relevant_context = rag.search_relevant_context(message, k=5) if rag else ""
        print(f"✓ Retrieved {len(relevant_context)} characters of knowledge-base context")

        previous_questions = get_previous_user_questions(user_id, max_items=4)

        # Final Answer - Context-aware, structured, and sourced
        final_answer_prompt = build_contextual_answer_prompt(
            message,
            relevant_context,
            personal_info,
            previous_questions=previous_questions,
            conversation_history=conversation_history,
            use_history_only=False,
        )
        ai_response = llm_client.generate(final_answer_prompt)

        # Belt-and-braces: strip any thinking artifacts and normalize the prose before it goes out
        ai_response = clean_llm_output(ai_response)
        ai_response = clean_prose_output(ai_response)
        append_session_history(user_id, 'assistant', ai_response)

        log_message(user_id, message, is_user=False, response=ai_response)

        return jsonify({
            'response': ai_response,
            'success': True,
            'session_id': user_id
        })

    except Exception as e:
        error_msg = f'Failed to get AI response: {str(e)}'
        user_id = get_user_id()
        log_message(user_id, message if message else 'Unknown', is_user=False, error=error_msg)
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to get AI response', 'success': False}), 500


@app.route('/api/ingest', methods=['POST'])
def ingest_knowledge():
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401

        uploaded_by = auth_user.get('email', 'admin')
        ingested_docs = []

        # Handle uploaded files
        uploaded_files = []
        uploaded_files.extend(request.files.getlist('files') or [])
        uploaded_files.extend(request.files.getlist('file') or [])
        uploaded_files.extend(request.files.getlist('files[]') or [])
        if not uploaded_files and 'file' in request.files:
            uploaded_files.append(request.files['file'])

        for uploaded_file in uploaded_files:
            if uploaded_file and uploaded_file.filename:
                try:
                    # Save uploaded file to disk first (knowledge_ingester reads from file)
                    safe_filename = secure_filename(uploaded_file.filename)
                    save_path = uploads_dir / safe_filename
                    uploaded_file.save(save_path)
                    try:
                        content = _read_local_file(str(save_path))
                    except RuntimeError as exc:
                        # Likely missing parser (e.g., pypdf), surface a helpful error
                        msg = f"Failed to extract text from {uploaded_file.filename}: {exc}"
                        print(f"⚠️ {msg}")
                        return jsonify({'error': msg, 'success': False}), 500

                    # Store extracted text in database
                    doc = store_document(
                        filename=uploaded_file.filename,
                        content=content,
                        source_type="file_upload",
                        uploaded_by=uploaded_by
                    )

                    if doc:
                        ingested_docs.append({
                            'filename': doc['filename'],
                            'source_type': doc['source_type'],
                            'id': doc['id']
                        })
                        print(f"✓ Stored document: {uploaded_file.filename}")
                except Exception as e:
                    print(f"⚠️ Failed to process file {uploaded_file.filename}: {e}")
                    return jsonify({'error': f'Failed to process {uploaded_file.filename}: {str(e)}', 'success': False}), 400

        # Handle URLs (if provided)
        data = request.get_json(silent=True) or {}
        urls = data.get('urls', []) if isinstance(data, dict) else []

        if not urls:
            urls.extend(request.form.getlist('urls') or [])

        if request.form.get('urls') and not urls:
            raw_urls = request.form.get('urls', '')
            urls.extend([u.strip() for u in raw_urls.split(',') if u.strip()])

        for url in urls:
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                content = response.text

                doc = store_document(
                    filename=url,
                    content=content,
                    source_type="url",
                    uploaded_by=uploaded_by
                )

                if doc:
                    ingested_docs.append({
                        'filename': url,
                        'source_type': 'url',
                        'id': doc['id']
                    })
                    print(f"✓ Stored URL content: {url}")
            except Exception as e:
                print(f"⚠️ Failed to fetch URL {url}: {e}")
                return jsonify({'error': f'Failed to fetch {url}: {str(e)}', 'success': False}), 400

        if not ingested_docs:
            return jsonify({'error': 'No files or URLs provided or the files were already ingested', 'success': False}), 400

        # Rebuild RAG with new documents
        rag = get_rag_system()
        if rag:
            try:
                print("🔄 Rebuilding RAG system with uploaded documents...")
                rag.build_vectorstore()  # This will now include document content
            except Exception as e:
                print(f"⚠️ RAG rebuild warning: {e}")
                traceback.print_exc()

        return jsonify({
            'success': True,
            'message': f'Successfully ingested {len(ingested_docs)} source(s)',
            'count': len(ingested_docs),
            'sources': ingested_docs
        })
    except Exception as e:
        error_message = str(e)
        print(f"Error ingesting knowledge: {error_message}")
        traceback.print_exc()
        return jsonify({'error': error_message, 'success': False}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents."""
    try:
        auth_user = require_auth()
        if not auth_user:
            return jsonify({'error': 'Authentication required', 'success': False}), 401

        from db_utils import get_all_documents
        documents = get_all_documents()

        # Remove full content for list view, just send metadata
        doc_list = [
            {
                'id': doc['id'],
                'filename': doc['filename'],
                'source_type': doc['source_type'],
                'created_at': doc['created_at'],
                'content_preview': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content']
            }
            for doc in documents
        ]

        return jsonify({
            'success': True,
            'count': len(doc_list),
            'documents': doc_list
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete an uploaded document."""
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401

        from db_utils import delete_document
        success = delete_document(doc_id)

        if success:
            # Rebuild RAG without the deleted document
            rag = get_rag_system()
            if rag:
                try:
                    print("🔄 Rebuilding RAG system after document deletion...")
                    rag.build_vectorstore()
                except Exception as e:
                    print(f"⚠️ RAG rebuild warning: {e}")
                    traceback.print_exc()

            return jsonify({'success': True, 'message': 'Document deleted successfully'})
        else:
            return jsonify({'error': 'Document not found', 'success': False}), 404
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/rebuild_rag', methods=['POST'])
def rebuild_rag():
    """Admin endpoint to rebuild the RAG vectorstore (re-index uploaded chunks)."""
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401

        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not initialized', 'success': False}), 500

        rag.build_vectorstore()
        return jsonify({'success': True, 'message': 'RAG rebuild triggered'})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    rag_status = "initialized" if rag_system else "not initialized"
    return jsonify({
        'status': 'healthy',
        'provider': LLM_PROVIDER,
        'llm_model': llm_client.config.model,
        'llm_base_url': llm_client.config.base_url,
        'llm_configured': llm_client.is_configured,
        'rag_system': rag_status,
        'rag_vectorstore_ready': rag_system is not None and rag_system.vectorstore is not None,
        'rag_keyword_fallback': rag_system is not None and rag_system.vectorstore is None
    })


@app.route('/api/rebuild-vectorstore', methods=['POST'])
def rebuild_vectorstore():
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401
        global rag_system
        rag_system = RAGSystem()
        rag_system.build_vectorstore()
        return jsonify({'message': 'Vector database rebuilt', 'success': True})
    except Exception as e:
        print(f"Error rebuilding vectorstore: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to rebuild vector database', 'success': False}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    print(f"🚀 Starting AI Assistant Backend with RAG ({LLM_PROVIDER.capitalize()})...")
    print(f"🤖 Model: {llm_client.config.model} at {llm_client.config.base_url}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")
    print(f"🌐 Server running at: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
