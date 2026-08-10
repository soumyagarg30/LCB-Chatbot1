import os
import time
import hmac
import hashlib
import base64
import json
from typing import Any, Dict, Optional


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data.encode("utf-8") + padding.encode("utf-8"))


def create_token(payload: Dict[str, Any], secret: str, expiry_seconds: int = 60 * 60 * 8) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    body = dict(payload)
    body["iat"] = now
    body["exp"] = now + expiry_seconds

    encoded_header = _b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    encoded_signature = _b64encode(signature)
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def verify_token(token: Optional[str], secret: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None

    parts = token.split(".")
    if len(parts) != 3:
        return None

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_signature = _b64encode(hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest())

    if not hmac.compare_digest(expected_signature, signature_b64):
        return None

    try:
        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    if payload.get("exp", 0) <= int(time.time()):
        return None

    return payload


def get_secret() -> str:
    return os.getenv("JWT_SECRET", "lcb-super-secret-key-2024")
