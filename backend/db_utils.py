import sqlite3
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib
import hmac

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "lcb_users.db"))


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            source_type TEXT NOT NULL,
            uploaded_by TEXT,
            file_hash TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )

    cursor.execute("SELECT email FROM users WHERE email = ?", ("admin@lcb.com",))
    if not cursor.fetchone():
        admin_hash = hash_password("LCB@1234")
        cursor.execute(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, ?, ?, ?)",
            ("admin@lcb.com", admin_hash, "Admin", "admin"),
        )

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hash_value: str) -> bool:
    return hmac.compare_digest(hash_password(password), hash_value)


def create_user(email: str, password: str, name: str) -> Optional[Dict[str, Any]]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, ?, ?, ?)",
            (email, password_hash, name, "user"),
        )
        conn.commit()
        cursor.execute("SELECT id, email, name, role FROM users WHERE email = ?", (email,))
        user = dict(cursor.fetchone())
        conn.close()
        return user
    except sqlite3.IntegrityError:
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password_hash, name, role FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def verify_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_user_by_email(email)
    if not user:
        return None

    if verify_password(password, user["password_hash"]):
        return {
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        }

    return None


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= length:
            break
        start = end - overlap
    return chunks


def store_document(filename: str, content: str, source_type: str = "upload", uploaded_by: str = "admin") -> Optional[Dict[str, Any]]:
    """Store a document in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return None
        
        cursor.execute(
            "INSERT INTO documents (filename, content, source_type, uploaded_by, file_hash) VALUES (?, ?, ?, ?, ?)",
            (filename, content, source_type, uploaded_by, file_hash),
        )
        conn.commit()
        cursor.execute("SELECT id, filename, source_type, created_at FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        doc = dict(row) if row else None

        # Create chunks and store them in document_chunks
        if doc:
            doc_id = doc['id']
            chunks = _chunk_text(content, chunk_size=1000, overlap=150)
            for idx, chunk in enumerate(chunks):
                cursor.execute(
                    "INSERT INTO document_chunks (document_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
                    (doc_id, idx, chunk),
                )
            conn.commit()
        conn.close()
        return doc
    except sqlite3.OperationalError as exc:
        if 'locked' in str(exc).lower():
            return None
        raise
    except sqlite3.IntegrityError:
        return None


def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieve all stored documents."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, content, source_type, created_at FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_document_chunks() -> List[Dict[str, Any]]:
    """Retrieve all document chunks with parent filename and metadata."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.id as chunk_id, c.document_id, c.chunk_index, c.chunk_text, d.filename, d.source_type, d.created_at
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY d.created_at DESC, c.chunk_index ASC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_documents_content() -> str:
    """Get all document content as a formatted string for RAG."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return ""
    
    content_parts = []
    for row in rows:
        content_parts.append(f"Source: {row['filename']}\n{row['content']}\n")
    
    return "\n---\n".join(content_parts)


def delete_document(doc_id: int) -> bool:
    """Delete a document from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Delete chunks first (ON DELETE CASCADE may not be enabled by default on some sqlite builds)
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
    except Exception:
        return False


def update_document_content(doc_id: int, new_content: str) -> bool:
    """Update the content of an existing document."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE documents SET content = ?, file_hash = ? WHERE id = ?", (
            new_content, hashlib.sha256(new_content.encode()).hexdigest(), doc_id
        ))
        # Remove existing chunks for this document and insert regenerated chunks
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
        chunks = _chunk_text(new_content, chunk_size=1000, overlap=150)
        for idx, chunk in enumerate(chunks):
            cursor.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
                (doc_id, idx, chunk),
            )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False
