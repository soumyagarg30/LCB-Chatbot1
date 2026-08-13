from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os
import json
import re
import html
import threading
import traceback
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from datetime import datetime
from dataclasses import replace
import uuid
from auth_utils import create_token, verify_token, get_secret
from db_utils import init_db, create_user, verify_user, store_document, get_documents_content, store_chat_message
from knowledge_ingester import _read_local_file
from llm_client import LLMClient, LLMConfig
from judge_agent import LLMJudgeAgent
from supervisor_agent import SupervisorAgent

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
evaluation_csv_path = logs_dir / 'chatbot_evaluations.csv'
evaluation_csv_lock = threading.Lock()

EVALUATION_CSV_HEADERS = [
    'Timestamp',
    'User ID',
    'Agent',
    'Question',
    'Expected Answer',
    'Chatbot Result',
    'ChatGPT Result',
    'Relevance Score',
    'Correctness Score',
    'Groundedness Score',
    'Clarity Score',
    'Safety Score',
    'Overall Score',
    'Accuracy Gap',
    'Improved Answer',
]

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


def _csv_safe_text(value):
    """Prevent spreadsheet formula execution when a CSV is opened."""
    text = '' if value is None else str(value)
    if text.startswith(('=', '+', '-', '@')):
        return "'" + text
    return text


def save_chat_evaluation(
    user_id,
    question,
    answer,
    judgment=None,
    agent='chatbot',
    chatgpt_result='',
    improved_answer='',
):
    """Append one completed chatbot exchange to the evaluation CSV."""
    judgment = judgment or {}
    dimensions = judgment.get('dimensions') or {}
    missing = judgment.get('missing_information') or []
    feedback = (judgment.get('feedback') or '').strip()
    improvement_parts = [feedback] if feedback else []
    if missing:
        improvement_parts.append('Add: ' + '; '.join(str(item) for item in missing))

    correctness = dimensions.get('correctness', '')
    accuracy_gap = ''
    if isinstance(correctness, (int, float)):
        accuracy_gap = max(0, 100 - correctness)

    row = [
        datetime.now().isoformat(),
        user_id,
        agent,
        question,
        chatgpt_result,
        answer,
        '',
        dimensions.get('relevance', ''),
        correctness,
        dimensions.get('groundedness', ''),
        dimensions.get('clarity', ''),
        dimensions.get('safety', ''),
        judgment.get('score', ''),
        accuracy_gap,
        improved_answer or ' '.join(improvement_parts),
    ]

    try:
        with evaluation_csv_lock:
            needs_header = not evaluation_csv_path.exists() or evaluation_csv_path.stat().st_size == 0
            with evaluation_csv_path.open('a', newline='', encoding='utf-8-sig') as csv_file:
                writer = csv.writer(csv_file)
                if needs_header:
                    writer.writerow(EVALUATION_CSV_HEADERS)
                writer.writerow([_csv_safe_text(value) for value in row])
    except OSError as exc:
        logging.error('Failed to save chatbot evaluation CSV: %s', exc)


def generate_chatgpt_reference(question, context='', chatbot_answer=''):
    """Generate an independent OpenAI reference answer when configured."""
    if reference_llm_client is None:
        return ''

    context = (context or 'No reference context was supplied.')[:5000]
    prompt = f"""Write the ideal answer to the user question below.

Use the reference context when it is relevant. Correct omissions or unclear claims in
the existing chatbot answer. Return only the improved final answer—no analysis,
grading, preamble, or JSON.

USER QUESTION:
{question[:1500]}

REFERENCE CONTEXT:
{context}

EXISTING CHATBOT ANSWER:
{chatbot_answer[:5000]}
"""
    try:
        return normalize_markdown_response(reference_llm_client.generate(prompt))
    except Exception as exc:
        logging.warning('OpenAI reference answer generation failed: %s', exc)
        return ''


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


def normalize_markdown_response(text: str) -> str:
    """Unwrap providers that return JSON or escaped Markdown inside content."""
    text = clean_llm_output(text)
    for _ in range(2):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            break
        if isinstance(payload, str):
            text = payload.strip()
            continue
        if isinstance(payload, dict):
            choices = payload.get('choices') or []
            candidate = ((choices[0].get('message') or {}).get('content') if choices else None)
            candidate = candidate or payload.get('response') or payload.get('answer') or payload.get('output') or payload.get('text')
            if isinstance(candidate, str):
                text = candidate.strip()
                continue
        break
    # Some OpenAI-compatible gateways send a JSON-escaped string as content.
    if '\\n' in text and '\n' not in text:
        text = text.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\t', '    ').replace('\\"', '"').replace("\\'", "'")
    return clean_llm_output(text)


def build_source_summary(sources: list[dict]) -> str:
    """Create a short source summary for the model prompt."""
    if not sources:
        return ""
    labels = []
    seen = set()
    for source in sources:
        label = format_source_provenance(source)

        if label and label not in seen:
            labels.append(label)
            seen.add(label)
        if len(labels) >= 6:
            break
    if not labels:
        return ""
    summary_lines = ["SOURCE SUMMARY:",
                     "- Use these source labels to form the final source attribution sentence in your answer."]
    summary_lines.extend([f"- {label}" for label in labels])
    if len(sources) > len(labels):
        summary_lines.append("- Additional relevant sources are available.")
    return "\n".join(summary_lines) + "\n\n"


def build_precise_attribution(sources: list[dict]) -> str:
    """Build exact record/chunk provenance for the retrieved evidence."""
    if not sources:
        return ""
    details = []
    for source in sources:
        detail = format_source_provenance(source)
        if detail and detail not in details:
            details.append(detail)
        if len(details) >= 5:
            break
    return "Sources used: " + "; ".join(details) + "." if details else ""


def format_source_provenance(source: dict) -> str:
    """Render a human-readable database record or uploaded chunk location."""
    source_type = (source.get('source_type') or '').lower()
    label = (source.get('label') or '').strip()
    record_type = (source.get('record_type') or '').strip().lower()

    if source_type in ('brand_kb', 'brand knowledge base'):
        if record_type == 'mechanism':
            return 'Brand Knowledge Base — Microbial Mechanism & Functions record'
        if record_type == 'faq' and source.get('faq_id') is not None:
            return f"Brand Knowledge Base — FAQ #{source['faq_id']}"
        if record_type == 'product' and source.get('crop'):
            return f"Brand Knowledge Base — {source['crop']} product record"
        if record_type == 'brand_overview':
            return 'Brand Knowledge Base — brand overview record'
        return f"Brand Knowledge Base — {record_type.replace('_', ' ')} record" if record_type else 'Brand Knowledge Base'

    if source_type in ('uploaded_document', 'file_upload', 'uploaded', 'uploaded_document_chunk'):
        filename = re.sub(r'^Uploaded Document:\s*', '', label) or 'uploaded document'
        return f"Uploaded document: {filename}"

    if source_type in ('url', 'website', 'web', 'ingested_source'):
        return f"Website source: {label or 'website content'}"
    return label or str(source.get('source_type', 'Knowledge source'))


def ensure_precise_attribution(text: str, sources: list[dict]) -> str:
    """Ensure the model output contains a precise attribution sentence."""
    if not text or not sources:
        return text

    attribution = build_precise_attribution(sources)
    if not attribution:
        return text

    # Replace any existing generic attribution line with the precise one.
    new_text = re.sub(
        r'(?im)^.*(?:this answer is based on|sources used:).*$',
        attribution,
        text.strip(),
        count=1
    )

    if new_text != text.strip():
        return new_text.strip()

    # Append attribution if not present.
    return text.strip() + "\n\n" + attribution


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


def build_tracker_answer(question: str, plans: list[dict]) -> str:
    """Answer tracker questions from the saved marketing plans stored in SQLite."""
    normalized_question = (question or "").strip().lower()
    if not plans:
        return "There are no plans in the tracker yet, so I do not have any strategy data to answer from."

    if not normalized_question:
        return "Please ask a question about the tracker, such as strategy, owner, status, or an overview."

    if any(term in normalized_question for term in ["strategy", "strategies", "plan", "plans"]):
        lines = [f"- {plan.get('title', 'Untitled plan')} — {plan.get('strategy', 'No strategy text saved')}" for plan in plans]
        return "Here are the strategies currently saved in the tracker:\n" + "\n".join(lines)

    if any(term in normalized_question for term in ["owner", "assigned", "who", "person"]):
        lines = [f"- {plan.get('title', 'Untitled plan')} → {plan.get('owner') or 'Unassigned'}" for plan in plans]
        return "Here is the current owner information from the tracker:\n" + "\n".join(lines)

    if any(term in normalized_question for term in ["status", "progress", "complete", "not started", "started"]):
        lines = [f"- {plan.get('title', 'Untitled plan')} → {plan.get('status', 'not_started')}" for plan in plans]
        return "Here is the current status snapshot from the tracker:\n" + "\n".join(lines)

    if any(term in normalized_question for term in ["summary", "overview", "what is in the tracker", "show me", "all"]):
        lines = [f"- {plan.get('title', 'Untitled plan')} ({plan.get('status', 'not_started')}) — owner: {plan.get('owner') or 'Unassigned'}" for plan in plans]
        return "Here is a quick tracker overview:\n" + "\n".join(lines)

    matches = []
    for plan in plans:
        haystack = " ".join([
            str(plan.get("title", "")),
            str(plan.get("strategy", "")),
            str(plan.get("owner", "")),
            str(plan.get("status", "")),
        ]).lower()
        if normalized_question in haystack:
            matches.append(plan)

    if matches:
        lines = [f"- {plan.get('title', 'Untitled plan')} — {plan.get('strategy', 'No strategy text saved')}" for plan in matches]
        return "I found relevant tracker entries:\n" + "\n".join(lines)

    return "I can answer questions about strategy, owner, status, and overall tracker progress. Here is a quick overview:\n" + "\n".join(
        f"- {plan.get('title', 'Untitled plan')} ({plan.get('status', 'not_started')})" for plan in plans
    )


def extract_website_text(content: str) -> str:
    """Convert fetched HTML into readable text before it enters the knowledge base."""
    content = re.sub(r'<(script|style|noscript|svg)[^>]*>.*?</\1>', ' ', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'</(p|div|section|article|li|h[1-6]|tr)>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<[^>]+>', ' ', content)
    content = html.unescape(content)
    content = re.sub(r'[ \t]+', ' ', content)
    content = re.sub(r'\n\s*\n+', '\n', content)
    return content.strip()


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
    source_summary: str | None = None,
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

    source_summary_section = (source_summary.strip() + "\n\n") if source_summary else ""

    from db_utils import get_agent_prompt
    prompt_entry = get_agent_prompt('contextual_answer')
    if prompt_entry:
        template = prompt_entry['prompt_text']
    else:
        prompt_path = Path(__file__).resolve().parent / 'prompts' / 'contextual_answer_prompt.txt'
        template = prompt_path.read_text(encoding='utf-8')

    return template.format(
        brand_name=brand_name,
        user_question=message,
        retrieved_context=context,
        previous_questions_section=previous_questions_section,
        source_summary_section=source_summary_section,
        clarification_guidance=guidance,
        context_label=context_label,
    )


def _create_retry_session(retries: int = 3, backoff_factor: float = 1.0, status_forcelist=None) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist or [429, 500, 502, 503, 504],
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _fetch_website_text(url: str) -> str:
    """Fetch a URL and return readable extracted text."""
    session = _create_retry_session()
    browser_headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'DNT': '1',
        'Connection': 'keep-alive',
    }

    def fetch_with_headers(headers: dict) -> requests.Response:
        response = session.get(
            url,
            timeout=15,
            headers=headers,
            allow_redirects=True,
        )
        if response.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{response.status_code} {response.reason}", response=response
            )
        return response

    try:
        response = fetch_with_headers(browser_headers)
    except requests.exceptions.HTTPError as first_exc:
        if first_exc.response is not None and first_exc.response.status_code == 503:
            fallback_headers = browser_headers.copy()
            fallback_headers['Cache-Control'] = 'max-age=0'
            try:
                response = fetch_with_headers(fallback_headers)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 'unknown'
                reason = exc.response.reason if exc.response is not None else 'Unknown'
                raise RuntimeError(
                    f"Unable to fetch website content; remote site returned {status} {reason} for url: {url}."
                ) from exc
        else:
            raise RuntimeError(f"Failed to fetch website URL: {first_exc}") from first_exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to fetch website URL: {exc}") from exc

    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        text = extract_website_text(response.text)
    elif 'text/plain' in content_type:
        text = response.text.strip()
    else:
        raise RuntimeError(f'Unsupported URL content type: {content_type}')

    if not text:
        raise RuntimeError('Unable to extract readable text from the website URL.')
    return text


def build_website_assessment_prompt(url: str, website_text: str) -> str:
    from db_utils import get_agent_prompt
    prompt_entry = get_agent_prompt('website_assessment')
    if prompt_entry:
        template = prompt_entry['prompt_text']
    else:
        prompt_path = Path(__file__).resolve().parent / 'prompts' / 'website_assessment_prompt.txt'
        template = prompt_path.read_text(encoding='utf-8')

    trimmed_text = website_text.strip()
    if len(trimmed_text) > 8000:
        trimmed_text = trimmed_text[:8000].rsplit(' ', 1)[0] + '...'
    return template.format(url=url, website_text=trimmed_text)


def assess_website_text(url: str, website_text: str) -> str:
    prompt = build_website_assessment_prompt(url, website_text)
    response = llm_client.generate(prompt)
    return normalize_markdown_response(response)


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

    # Preserve the structured mechanism list when a provider returns an empty
    # answer. Flattening the context into prose would otherwise discard exactly
    # the microorganism/function detail the user requested.
    if re.search(r"\b(microbe|microbial|microorganisms?|mechanisms?)\b", message or "", re.IGNORECASE):
        mechanism_pairs = re.findall(
            r"^[ \t]*-[ \t]*([^\n-][^\n]*?)[ \t]+-[ \t]+([^\n]+?)[ \t]*$",
            relevant_context or "",
            flags=re.MULTILINE,
        )
        if mechanism_pairs:
            lines = ["The named microorganisms and their functions are:"]
            lines.extend(f"- {name.strip()}: {function.strip()}" for name, function in mechanism_pairs)
            lines.append("This answer is based on the Brand Knowledge Base.")
            return "\n".join(lines)

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
    if dedup_sources:
        evidence_note = f"This response is drawn from {', '.join(dedup_sources[:3])}."

    lines = [answer_sent]
    for detail in detail_parts:
        lines.append(detail)
    if dedup_sources:
        lines.append(f"Sources: {', '.join(dedup_sources[:3])}.")
    lines.append(evidence_note)

    return clean_prose_output("\n\n".join(lines))


# Default is local Ollama with llama3.2:3b. See env.example for other providers.
llm_client = LLMClient()
# Competitive reports should stay concise on the local model. A smaller output
# budget materially reduces response time while still allowing a useful report.
competitive_llm_client = LLMClient(replace(llm_client.config, timeout=75, max_tokens=900))
# Reasoning providers can consume a large part of the output budget before
# emitting their final JSON, so the judge gets a larger dedicated allowance.
judge_llm_client = LLMClient(
    replace(llm_client.config, temperature=0, max_tokens=max(llm_client.config.max_tokens, 4096))
)
judge_agent = LLMJudgeAgent(client=judge_llm_client)

openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
reference_llm_client = None
if openai_api_key:
    reference_llm_client = LLMClient(LLMConfig(
        provider='openai_compatible',
        model=os.getenv('OPENAI_REFERENCE_MODEL', 'gpt-4o-mini'),
        base_url=os.getenv('OPENAI_REFERENCE_BASE_URL', 'https://api.openai.com/v1'),
        api_key=openai_api_key,
        timeout=int(os.getenv('OPENAI_REFERENCE_TIMEOUT', '120')),
        temperature=0.2,
        max_tokens=int(os.getenv('OPENAI_REFERENCE_MAX_TOKENS', '1600')),
    ))
supervisor_agent = SupervisorAgent()
LLM_PROVIDER = llm_client.config.provider

COMPETITIVE_MARKET_BASELINE = """
EXTERNAL COMPANY CANDIDATES (official company sources; verify local overlap before calling any company a direct competitor):
- Coromandel International Limited — an external company offering organic fertilisers, microbial bio-fertilizers, specialty nutrients, crop protection, farm advisory, and a large rural retail network. Official sources: https://www.coromandel.biz/company/ and https://www.coromandel.biz/products-services/organic/
- Indian Farmers Fertiliser Cooperative Limited (IFFCO) — an external cooperative offering organic and bio-fertilisers including Rhizobium, Azotobacter, Acetobacter, PSB, KMB, ZSB, NPK liquid consortia, and Sagarika products. Official source: https://www.iffco.in/en/organic-and-bio-fertilisers
- Krishak Bharati Cooperative Limited (KRIBHCO) — an external cooperative whose portfolio includes bio-fertilizers, compost, natural potash, and other fertilisers distributed through channel partners and Krishak Bharati Sewa Kendras. Official sources: https://kribhco.net/ and https://www.kribhco.net/pages/products/product_org.html

ENTITY BOUNDARY:
- LCB Fertilizers is the subject company, never a competitor.
- Navyakosh is LCB's product/brand, never a company and never a competitor.
- LCB products, technologies, reports, projects, and internal brands belong only in the LCB side of a comparison.
- A competitor row must name a distinct external legal company or cooperative. Never use a product name as the competitor name.
""".strip()


def build_competitor_fallback(message: str) -> str:
    """Return an intent-aware, evidence-backed answer when local generation is slow."""
    text = (message or '').lower()

    if any(term in text for term in ('30/60/90', '30-60-90', '90-day', '90 day', 'action plan', 'roadmap', 'timeline')):
        return """## 30/60/90-day competitive action plan

### Days 1–30 — Establish the facts

- **Sales team:** interview 10–15 dealers in each priority district about Coromandel, IFFCO, and KRIBHCO availability, pack sizes, prices, retailer margins, credit terms, and farmer demand.
- **Field team:** collect competitor pack photographs and document the claims printed on them; do not rely on memory or hearsay.
- **Marketing:** create a district-level competitor matrix covering products, price per acre, dealer coverage, promotions, and evidence quality.
- **Management:** select two priority districts and approve baseline metrics: active dealers, repeat orders, demonstrations, leads, and conversion rate.

**Day-30 deliverable:** verified local competitor ranking and two district battlecards.

### Days 31–60 — Prove and differentiate

- **Agronomy/field team:** launch side-by-side Navyakosh demonstrations with consistent plots and documented soil, irrigation, input-cost, yield, and quality measures.
- **Marketing:** turn verified Navyakosh evidence into a one-page dealer comparison sheet and local-language farmer material. Avoid unsupported competitor claims.
- **Sales:** recruit or reactivate dealers in villages where candidate competitors have weak availability; give dealers product training and demonstration support.
- **Operations:** track stock availability and replenishment time so demand-generation activity is never unsupported by supply.

**Day-60 deliverable:** live demonstrations, trained pilot dealers, and an evidence-based sales kit.

### Days 61–90 — Scale what converts

- Compare dealer activation, farmer attendance, trial-to-purchase conversion, repeat orders, and revenue by district.
- Scale the best-performing demonstration and dealer program; stop activities that do not produce qualified trials or repeat sales.
- Publish only verified field results through farmer meetings, WhatsApp content, dealer counters, and local agronomy partners.
- Set the next-quarter target using observed conversion and replenishment capacity, not an unsupported market-share estimate.

**Day-90 deliverable:** a repeatable district playbook with named owners, budget, targets, and weekly reporting.

## Metrics

Dealer interviews completed; verified competitor SKUs; active/trained dealers; demonstrations launched; farmer attendance; qualified trials; trial-to-purchase conversion; repeat-order rate; stock-out rate; revenue per activated dealer.

This plan treats Coromandel International, IFFCO, and KRIBHCO as **candidate competitors** until direct local overlap is verified from dealer evidence."""

    if any(term in text for term in ('compare', 'comparison', 'versus', ' vs ', 'positioning', 'dealer network', 'farmer engagement')):
        return """## Evidence-based competitor comparison

| Company | Relevant overlap with Navyakosh | Visible strength | What LCB must verify locally |
|---|---|---|---|
| **Coromandel International** | Organic fertilisers and microbial bio-products | Broad crop-input portfolio, farm advisory, and rural retail network | Products, outlets, dealer terms, and farmer adoption in LCB priority districts |
| **IFFCO** | Organic and bio-fertilisers including microbial nutrient products | Cooperative reach, established farmer recognition, and broad bio-fertiliser range | Local SKU availability, price per acre, promotion, and dealer pull |
| **KRIBHCO** | Bio-fertilizers, compost, natural potash, and related inputs | Cooperative/channel distribution and farmer service centres | District coverage, retailer margin, product movement, and farmer preference |
| **LCB / Navyakosh** | Organic fertiliser positioned around soil health and water retention | Potential for focused local service, demonstrations, and faster market learning | Verified comparative field results, price per acre, repeat purchase, and dealer coverage |

## Strategic implication

LCB should not try to match larger firms portfolio-for-portfolio. It should compete through concentrated district coverage, responsive dealer support, and independently documented Navyakosh results.

## Immediate actions

1. Build a verified SKU-and-price comparison from local dealer visits.
2. Run standardized side-by-side demonstrations and record cost-per-acre and outcome data.
3. Create district battlecards using only validated claims.
4. Prioritize underserved villages where larger companies have weak product availability.

Sources: [Coromandel](https://www.coromandel.biz/company/), [IFFCO](https://www.iffco.in/en/organic-and-bio-fertilisers), and [KRIBHCO](https://kribhco.net/). These are candidate competitors until local overlap is confirmed."""

    if any(term in text for term in ('opportunit', 'get ahead', 'outperform', 'beat', 'advantage', 'strategy')):
        return """## Three best opportunities to get ahead

1. **Own selected districts instead of spreading thinly.** Map competitor availability dealer by dealer, then build dependable Navyakosh stock, training, and field support in underserved clusters.
2. **Turn product claims into comparative proof.** Run consistent side-by-side demonstrations and publish verified cost-per-acre, soil, irrigation, yield, and repeat-purchase results. Evidence is more defensible than broad promotional claims.
3. **Make dealers successful with Navyakosh.** Combine competitive margins, rapid replenishment, local-language selling tools, farmer meetings, and agronomy support. Track active dealers and repeat orders rather than registrations alone.

## Competitive threats to monitor

- Coromandel's broad portfolio, advisory capability, and retail reach.
- IFFCO's cooperative distribution and established bio-fertiliser range.
- KRIBHCO's channel network, service centres, and related organic inputs.

## Start this month

- Interview 10–15 dealers per priority district.
- Record verified competitor SKUs, prices, margins, claims, and availability.
- Select two districts for Navyakosh comparison plots.
- Review dealer activation, farmer trials, conversions, repeat orders, and stock-outs weekly.

These firms are **candidate competitors**; rank them only after confirming direct local overlap. Sources: [Coromandel](https://www.coromandel.biz/company/), [IFFCO](https://www.iffco.in/en/organic-and-bio-fertilisers), and [KRIBHCO](https://kribhco.net/)."""

    return """## Candidate competitor companies

1. **Coromandel International Limited** — a strong candidate in organic fertilisers, microbial bio-fertilizers, specialty nutrients, crop protection, farm advisory, and rural retail. Its broad portfolio and distribution reach make it an important benchmark for LCB. [Official company profile](https://www.coromandel.biz/company/)
2. **Indian Farmers Fertiliser Cooperative Limited (IFFCO)** — offers organic and bio-fertilisers including Rhizobium, Azotobacter, PSB, KMB, ZSB, NPK liquid consortia, and Sagarika products. Its established cooperative distribution and farmer recognition make it a relevant candidate competitor. [Official product page](https://www.iffco.in/en/organic-and-bio-fertilisers)
3. **Krishak Bharati Cooperative Limited (KRIBHCO)** — offers bio-fertilizers, compost, natural potash, and other crop inputs through channel partners and its service centres. Its product-category and farmer-channel overlap make it another relevant candidate. [Official website](https://kribhco.net/)

These are **candidate competitors**, not yet proven direct local rivals. Confirm their presence, dealer coverage, pack sizes, and price points in LCB's target districts before ranking them.

## How LCB can get ahead

- **Win locally:** map dealers and product availability for these companies in each priority district, then fill underserved dealer and village gaps.
- **Prove Navyakosh's value:** run comparable field demonstrations with documented soil, irrigation, yield, and cost-per-acre results.
- **Make the offer easy to compare:** create a one-page dealer sheet showing verified use cases, application rate, farmer economics, and evidence—without unsupported competitor claims.
- **Build dealer preference:** provide training, demonstration support, rapid replenishment, and farmer-meeting materials in local languages.

## Next validation step

Collect competitor product photos, dealer quotations, pack sizes, retailer margins, and availability from 10–15 dealers in each priority district. That evidence will turn this candidate list into a precise local competitor ranking."""


def is_competitor_identification_question(message: str) -> bool:
    text = (message or '').lower()
    identity_terms = ('who are', 'what are', 'main competitor', 'top competitor', 'name competitor', 'list competitor')
    return 'competitor' in text and any(term in text for term in identity_terms)


def has_fast_competitor_answer(message: str) -> bool:
    text = (message or '').lower()
    fast_terms = (
        '30/60/90', '30-60-90', '90-day', '90 day', 'action plan', 'roadmap',
        'compare', 'comparison', 'versus', ' vs ', 'positioning', 'dealer network',
        'farmer engagement', 'opportunit', 'get ahead', 'outperform', 'beat',
        'advantage', 'strategy',
    )
    return is_competitor_identification_question(message) or any(term in text for term in fast_terms)


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
        rag_system.prepare_in_memory_docs()
        return rag_system
    except Exception as e:
        print(f"⚠️ RAG initialization failed: {e}")
        traceback.print_exc()
        rag_system = None
        return None


# Initialize RAG eagerly so startup status reflects actual availability.
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

    store_chat_message(f"auth:{user['email']}", "system", f"signup:{user['email']}", user_email=user['email'])
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

    store_chat_message(f"auth:{user['email']}", "system", f"login:{user['email']}", user_email=user['email'])
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
        store_chat_message(user_id, 'user', message, user_email=auth_user.get('sub'))
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
        relevant_context = ""
        source_summary = ""
        source_info = []
        if rag:
            relevant_context, source_info = rag.search_relevant_context(message, k=5, return_sources=True)
            source_summary = build_source_summary(source_info)
        print(f"✓ Retrieved {len(relevant_context)} characters of knowledge-base context")

        previous_questions = get_previous_user_questions(user_id, max_items=4)

        routing = supervisor_agent.route(message, is_admin=auth_user.get('role') == 'admin')
        judgment_context = relevant_context

        if routing['agent'] == 'marketing':
            from db_utils import get_agent_prompt
            prompt_entry = get_agent_prompt('marketing_strategist')
            if prompt_entry:
                final_answer_prompt = prompt_entry['prompt_text']
            else:
                final_answer_prompt = (BASE_DIR / 'prompts' / 'marketing_strategist_prompt.txt').read_text(encoding='utf-8')
            final_answer_prompt = final_answer_prompt.format(
                relevant_context=relevant_context or 'No matching internal source was found. State assumptions and suggest what to validate.',
                message=message,
            )
            ai_response = llm_client.generate(final_answer_prompt)
        elif routing['agent'] == 'competitive_intelligence':
            from db_utils import get_agent_prompt
            prompt_entry = get_agent_prompt('competitive_intelligence')
            if prompt_entry:
                final_answer_prompt = prompt_entry['prompt_text']
            else:
                final_answer_prompt = (BASE_DIR / 'prompts' / 'competitive_intelligence_prompt.txt').read_text(encoding='utf-8')
            final_answer_prompt = final_answer_prompt.format(
                relevant_context=relevant_context or 'No matching competitor evidence was found. State assumptions, identify the evidence needed, and provide only a preliminary framework.',
                message=message,
            )
            ai_response = llm_client.generate(final_answer_prompt)
        elif routing['agent'] == 'tracker':
            from db_utils import get_marketing_plans
            tracker_plans = get_marketing_plans()
            ai_response = build_tracker_answer(message, tracker_plans)
            judgment_context = json.dumps(tracker_plans, ensure_ascii=False)
        else:
            final_answer_prompt = build_contextual_answer_prompt(
                message,
                relevant_context,
                personal_info,
                previous_questions=previous_questions,
                conversation_history=conversation_history,
                source_summary=source_summary,
                use_history_only=False,
            )
            try:
                ai_response = llm_client.generate(final_answer_prompt)
            except RuntimeError as exc:
                logging.warning(
                    "General answer generation unavailable; using grounded local fallback: %s",
                    exc,
                )
                ai_response = synthesize_local_answer(message, relevant_context, personal_info)

        # Belt-and-braces: strip any thinking artifacts and normalize the prose before it goes out
        ai_response = clean_llm_output(ai_response)
        ai_response = clean_prose_output(ai_response)
        ai_response = ensure_precise_attribution(ai_response, source_info)
        judgment = judge_agent.judge(
            question=message,
            response=ai_response,
            context=judgment_context,
            agent_name=f"{routing['agent']}_chatbot",
        )
        chatgpt_result = generate_chatgpt_reference(message, judgment_context, ai_response)
        logging.info("LLM judge result for %s: %s", user_id, json.dumps(judgment, ensure_ascii=False))
        append_session_history(user_id, 'assistant', ai_response)
        store_chat_message(user_id, 'assistant', ai_response, user_email=auth_user.get('sub'))

        log_message(user_id, message, is_user=False, response=ai_response)
        save_chat_evaluation(
            user_id,
            message,
            ai_response,
            judgment,
            agent=routing['display_name'],
            chatgpt_result=chatgpt_result,
            improved_answer=chatgpt_result,
        )

        return jsonify({
            'response': ai_response,
            'judgment': judgment,
            'active_agent': routing['display_name'],
            'routing': routing,
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


@app.route('/api/tracker/chat', methods=['POST'])
def tracker_chat():
    auth_user = require_auth(required_role='admin')
    if not auth_user:
        return jsonify({'error': 'Admin authentication required', 'success': False}), 401

    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({'error': 'A question is required', 'success': False}), 400

    from db_utils import get_marketing_plans
    plans = get_marketing_plans()
    answer = build_tracker_answer(question, plans)
    judgment = judge_agent.judge(
        question=question,
        response=answer,
        context=json.dumps(plans, ensure_ascii=False),
        agent_name='tracker_chatbot',
    )
    chatgpt_result = generate_chatgpt_reference(question, json.dumps(plans, ensure_ascii=False), answer)
    save_chat_evaluation(
        auth_user.get('sub', 'admin'), question, answer, judgment,
        agent='Tracker', chatgpt_result=chatgpt_result, improved_answer=chatgpt_result,
    )
    return jsonify({'success': True, 'response': answer, 'judgment': judgment})


@app.route('/api/marketing/chat', methods=['POST'])
def marketing_chat():
    """Marketing strategist that is grounded in the same shared RAG knowledge base."""
    try:
        auth_user = require_auth()
        if not auth_user:
            return jsonify({'error': 'Authentication required', 'success': False}), 401
        data = request.get_json(silent=True) or {}
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message is required', 'success': False}), 400

        rag = get_rag_system()
        relevant_context = rag.search_relevant_context(message, k=6) if rag else ''
        from db_utils import get_agent_prompt
        prompt_entry = get_agent_prompt('marketing_strategist')
        if prompt_entry:
            prompt_template = prompt_entry['prompt_text']
        else:
            prompt_path = Path(__file__).resolve().parent / 'prompts' / 'marketing_strategist_prompt.txt'
            prompt_template = prompt_path.read_text(encoding='utf-8')
        prompt_template = prompt_template.format(
            relevant_context=relevant_context or 'No matching internal source was found. State assumptions and suggest what to validate.',
            message=message,
        )
        response = normalize_markdown_response(llm_client.generate(prompt_template))
        judgment = judge_agent.judge(
            question=message,
            response=response,
            context=relevant_context,
            agent_name='marketing_chatbot',
        )
        chatgpt_result = generate_chatgpt_reference(message, relevant_context, response)
        save_chat_evaluation(
            auth_user.get('sub', 'unknown'), message, response, judgment,
            agent='Marketing Strategist', chatgpt_result=chatgpt_result, improved_answer=chatgpt_result,
        )
        return jsonify({'success': True, 'response': response, 'judgment': judgment})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/competitors/chat', methods=['POST'])
def competitors_chat():
    """Dedicated competitive-intelligence chat grounded in the shared RAG knowledge base."""
    try:
        auth_user = require_auth()
        if not auth_user:
            return jsonify({'error': 'Authentication required', 'success': False}), 401
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'error': 'Message is required', 'success': False}), 400

        # The common identity question has a verified answer and should not wait
        # for local model generation.
        if has_fast_competitor_answer(message):
            response = build_competitor_fallback(message)
            judgment = judge_agent.judge_without_llm(
                question=message,
                response=response,
                context=COMPETITIVE_MARKET_BASELINE,
                agent_name='competitive_intelligence_chatbot',
            )
            chatgpt_result = generate_chatgpt_reference(message, COMPETITIVE_MARKET_BASELINE, response)
            save_chat_evaluation(
                auth_user.get('sub', 'unknown'),
                message,
                response,
                judgment,
                agent='Competitive Intelligence Strategist',
                chatgpt_result=chatgpt_result,
                improved_answer=chatgpt_result,
            )
            return jsonify({
                'success': True,
                'response': response,
                'judgment': judgment,
                'active_agent': 'Competitive Intelligence Strategist',
            })

        rag = get_rag_system()
        relevant_context = rag.search_relevant_context(message, k=4) if rag else ''
        from db_utils import get_agent_prompt
        prompt_entry = get_agent_prompt('competitive_intelligence')
        prompt_template = prompt_entry['prompt_text'] if prompt_entry else (BASE_DIR / 'prompts' / 'competitive_intelligence_prompt.txt').read_text(encoding='utf-8')
        evidence_context = COMPETITIVE_MARKET_BASELINE
        if relevant_context:
            evidence_context += f'\n\nLCB KNOWLEDGE-BASE CONTEXT:\n{relevant_context}'
        prompt = prompt_template.format(
            relevant_context=evidence_context,
            message=message,
        )
        try:
            response = normalize_markdown_response(competitive_llm_client.generate(prompt))
        except RuntimeError as exc:
            logging.warning('Competitive intelligence generation unavailable; using evidence-backed fallback: %s', exc)
            response = build_competitor_fallback(message)
        # Score every response without a second serial Ollama call.
        judgment = judge_agent.judge_without_llm(
            question=message,
            response=response,
            context=evidence_context,
            agent_name='competitive_intelligence_chatbot',
        )
        chatgpt_result = generate_chatgpt_reference(message, evidence_context, response)
        save_chat_evaluation(
            auth_user.get('sub', 'unknown'),
            message,
            response,
            judgment,
            agent='Competitive Intelligence Strategist',
            chatgpt_result=chatgpt_result,
            improved_answer=chatgpt_result,
        )
        return jsonify({'success': True, 'response': response, 'judgment': judgment, 'active_agent': 'Competitive Intelligence Strategist'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/marketing/plans', methods=['GET', 'POST'])
def marketing_plans():
    auth_user = require_auth()
    if not auth_user:
        return jsonify({'error': 'Authentication required', 'success': False}), 401
    from db_utils import create_marketing_plan, get_marketing_plans
    if request.method == 'GET':
        if auth_user.get('role') != 'admin':
            return jsonify({'error': 'Admin authentication required', 'success': False}), 403
        return jsonify({'success': True, 'plans': get_marketing_plans()})

    data = request.get_json(silent=True) or {}
    title, strategy = data.get('title', '').strip(), data.get('strategy', '').strip()
    if not title or not strategy:
        return jsonify({'error': 'A title and plan are required', 'success': False}), 400
    plan = create_marketing_plan(title, strategy, data.get('owner', '').strip(), auth_user.get('sub', 'unknown'))
    return jsonify({'success': True, 'plan': plan}), 201


@app.route('/api/marketing/plans/<int:plan_id>', methods=['PATCH'])
def update_marketing_plan(plan_id):
    auth_user = require_auth(required_role='admin')
    if not auth_user:
        return jsonify({'error': 'Admin authentication required', 'success': False}), 403
    data = request.get_json(silent=True) or {}
    status = data.get('status')
    if status not in ('not_started', 'in_progress', 'complete'):
        return jsonify({'error': 'Invalid status', 'success': False}), 400
    from db_utils import update_marketing_plan_status
    plan = update_marketing_plan_status(plan_id, status)
    if not plan:
        return jsonify({'error': 'Plan not found', 'success': False}), 404
    return jsonify({'success': True, 'plan': plan})


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
                response = requests.get(
                    url,
                    timeout=15,
                    headers={'User-Agent': 'LCB-Knowledge-Agent/1.0 (+marketing research)'},
                )
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    return jsonify({'error': f'URL must return a web page or text content: {url}', 'success': False}), 400
                content = extract_website_text(response.text) if 'text/html' in content_type else response.text.strip()
                if not content:
                    return jsonify({'error': f'No readable content found at: {url}', 'success': False}), 400

                doc = store_document(
                    filename=url,
                    content=content,
                    source_type="url",
                    uploaded_by=uploaded_by
                )

                if doc:
                    assessment = None
                    try:
                        assessment = assess_website_text(url, content)
                        print(f"✓ Assessed URL content: {url}")
                    except Exception as assess_exc:
                        print(f"⚠️ Website assessment failed for {url}: {assess_exc}")
                        assessment = str(assess_exc)

                    ingested_docs.append({
                        'filename': url,
                        'source_type': 'url',
                        'id': doc['id'],
                        'assessment': assessment,
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
        print(f"Error ingesting knowledge: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/assess-website', methods=['POST'])
def assess_website_route():
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401

        data = request.get_json(silent=True) or {}
        url = (data.get('url') or '').strip()
        if not url:
            return jsonify({'error': 'URL is required', 'success': False}), 400

        website_text = _fetch_website_text(url)
        assessment = assess_website_text(url, website_text)
        return jsonify({'success': True, 'url': url, 'assessment': assessment})
    except RuntimeError as e:
        print(f"Website fetch error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 502
    except Exception as e:
        print(f"Error assessing website: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


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


@app.route('/api/vectorstore-info', methods=['GET'])
def vectorstore_info():
    """Inspect the live RAG corpus, chunk metadata, and vector index status."""
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401

        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system is not initialized', 'success': False}), 503

        docs = rag.prepare_in_memory_docs()
        search = (request.args.get('search') or '').strip().lower()
        source_filter = (request.args.get('source_type') or '').strip().lower()
        page = max(1, request.args.get('page', default=1, type=int) or 1)
        page_size = min(100, max(1, request.args.get('page_size', default=25, type=int) or 25))

        all_chunks = []
        source_counts = {}
        total_characters = 0
        for index, doc in enumerate(docs):
            metadata = {
                str(key): value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
                for key, value in (doc.metadata or {}).items()
            }
            source_type = str(metadata.get('source_type') or metadata.get('type') or 'unknown')
            source_label = rag._document_source_label(doc)
            text = doc.page_content or ''
            total_characters += len(text)
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            item = {
                'id': index + 1,
                'chunk_index': metadata.get('chunk_index', index),
                'source_type': source_type,
                'source_label': source_label,
                'character_count': len(text),
                'word_count': len(text.split()),
                'content': text,
                'metadata': metadata,
            }
            haystack = f"{source_label} {source_type} {text} {json.dumps(metadata, ensure_ascii=False)}".lower()
            if search and search not in haystack:
                continue
            if source_filter and source_type.lower() != source_filter:
                continue
            all_chunks.append(item)

        filtered_count = len(all_chunks)
        start = (page - 1) * page_size
        paginated = all_chunks[start:start + page_size]
        chroma_count = None
        if rag.vectorstore is not None:
            try:
                chroma_count = rag.vectorstore._collection.count()
            except Exception:
                pass

        return jsonify({
            'success': True,
            'status': {
                'collection_name': rag.collection_name,
                'retrieval_mode': 'vector_mmr_with_keyword_blend' if rag.vectorstore is not None else 'keyword_fallback',
                'vectorstore_ready': rag.vectorstore is not None,
                'embedding_model': rag._embedding_model_name,
                'embeddings_initialized': rag._embeddings_initialized,
                'storage_path': str(rag.chroma_path),
                'persisted_vector_count': chroma_count,
            },
            'statistics': {
                'total_chunks': len(docs),
                'filtered_chunks': filtered_count,
                'total_characters': total_characters,
                'average_chunk_characters': round(total_characters / len(docs), 1) if docs else 0,
                'source_counts': source_counts,
            },
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_pages': max(1, (filtered_count + page_size - 1) // page_size),
            },
            'chunks': paginated,
        })
    except Exception as e:
        logging.exception('Failed to inspect vector database')
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/agent-prompts', methods=['GET'])
def list_agent_prompts():
    """List all agent prompt templates."""
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401

        from db_utils import get_all_agent_prompts
        prompts = get_all_agent_prompts()
        return jsonify({'success': True, 'prompts': prompts})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/agent-prompts/<agent_key>', methods=['GET', 'POST'])
def manage_agent_prompt(agent_key):
    """Retrieve or update a specific agent prompt."""
    try:
        auth_user = require_auth(required_role='admin')
        if not auth_user:
            return jsonify({'error': 'Admin authentication required', 'success': False}), 401

        from db_utils import get_agent_prompt, upsert_agent_prompt

        if request.method == 'GET':
            prompt = get_agent_prompt(agent_key)
            if not prompt:
                return jsonify({'error': 'Agent prompt not found', 'success': False}), 404
            return jsonify({'success': True, 'prompt': prompt})

        data = request.get_json(silent=True) or {}
        display_name = (data.get('display_name') or '').strip()
        prompt_text = (data.get('prompt_text') or '').strip()
        if not display_name or not prompt_text:
            return jsonify({'error': 'display_name and prompt_text are required', 'success': False}), 400
        required_placeholders = {
            'supervisor': ('{available_routes}', '{user_request}'),
            'llm_judge': ('{agent_name}', '{user_question}', '{reference_context}', '{chatbot_response}'),
        }
        missing = [
            placeholder for placeholder in required_placeholders.get(agent_key, ())
            if placeholder not in prompt_text
        ]
        if missing:
            return jsonify({
                'error': f"Prompt is missing required placeholders: {', '.join(missing)}",
                'success': False,
            }), 400

        prompt = upsert_agent_prompt(agent_key, display_name, prompt_text)
        return jsonify({'success': True, 'prompt': prompt})
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
        'llm_judge_enabled': judge_agent.enabled,
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
    debug_enabled = os.getenv('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes', 'on')
    print(f"🚀 Starting AI Assistant Backend with RAG ({LLM_PROVIDER.capitalize()})...")
    print(f"🤖 Model: {llm_client.config.model} at {llm_client.config.base_url}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")
    print(f"🌐 Server running at: http://localhost:{port}")
    app.run(debug=debug_enabled, host='0.0.0.0', port=port, use_reloader=debug_enabled)
