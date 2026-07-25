import os
import re
import html
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".html", ".csv", ".json", ".xml"}


def _read_local_file(path: str) -> str:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if file_path.is_dir():
        raise ValueError(f"Path is a directory: {path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pypdf is required for PDF ingestion") from exc
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return _clean_text("\n".join(pages))

    if suffix in {".doc", ".docx"}:
        try:
            from docx import Document as DocxDocument
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("python-docx is required for DOCX ingestion") from exc
        doc = DocxDocument(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text("\n".join(paragraphs))

    if suffix in SUPPORTED_TEXT_EXTENSIONS or suffix == ".htm":
        return _clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))

    return _clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_url_content(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed")
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return _clean_text(response.text)


def ingest_sources(files: Optional[List[str]] = None, urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Ingest local files and website URLs into a simple list of documents."""
    sources: List[Dict[str, Any]] = []

    for file_path in files or []:
        content = _read_local_file(file_path)
        sources.append({
            "source_type": "file",
            "source": file_path,
            "content": _clean_text(content),
        })

    for url in urls or []:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        content = _fetch_url_content(url)
        sources.append({
            "source_type": "url",
            "source": url,
            "content": content,
        })

    return sources
