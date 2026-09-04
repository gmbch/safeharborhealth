from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import sys
import secrets
import threading
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CognitoToken:
    access_token: str
    id_token: str
    expires_in: int


def load_cognito_config() -> dict[str, str]:
    candidates = []
    override = os.environ.get("SAFEHARBOR_COGNITO_CONFIG", "").strip()
    if override:
        candidates.append(Path(override))
    candidates.append(Path(sys.executable).resolve().parent / "cognito_config.json")
    candidates.append(Path(__file__).resolve().with_name("cognito_config.json"))
    bundle_root = getattr(sys, "_MEIPASS", "")
    if bundle_root:
        candidates.append(Path(str(bundle_root)) / "cognito_config.json")
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(key): str(value) for key, value in payload.items() if value is not None}
        except (OSError, ValueError):
            continue
    return {}


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def login_with_cognito(timeout_seconds: int = 300) -> CognitoToken:
    config = load_cognito_config()
    domain = (os.environ.get("SAFEHARBOR_COGNITO_HOSTED_UI") or config.get("hosted_ui_base_url", "")).strip().rstrip("/")
    client_id = (os.environ.get("SAFEHARBOR_COGNITO_CLIENT_ID") or config.get("client_id", "")).strip()
    redirect_uri = (
        os.environ.get("SAFEHARBOR_COGNITO_REDIRECT_URI")
        or config.get("redirect_uri", "http://127.0.0.1:8765/callback")
    ).strip()
    if not domain or not client_id:
        raise RuntimeError("Configure SAFEHARBOR_COGNITO_HOSTED_UI and SAFEHARBOR_COGNITO_CLIENT_ID.")

    parsed_redirect = urllib.parse.urlparse(redirect_uri)
    if parsed_redirect.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("Cognito redirect URI must use localhost for the desktop flow.")
    state = secrets.token_urlsafe(32)
    verifier = _base64url(secrets.token_bytes(48))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    result: dict[str, str] = {}
    done = threading.Event()

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if query.get("state", [""])[0] != state:
                result["error"] = "OAuth state validation failed"
            elif query.get("error"):
                result["error"] = query.get("error_description", query["error"])[0]
            else:
                result["code"] = query.get("code", [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body>You may close this window and return to SafeHarborAI.</body></html>")
            done.set()

        def log_message(self, *_args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer((parsed_redirect.hostname, parsed_redirect.port or 80), CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    params = {
        "client_id": client_id,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    authorize_url = f"{domain}/oauth2/authorize?{urllib.parse.urlencode(params)}"
    webbrowser.open(authorize_url)
    if not done.wait(timeout_seconds):
        server.server_close()
        raise TimeoutError("Timed out waiting for Cognito login.")
    server.server_close()
    if result.get("error"):
        raise RuntimeError(result["error"])
    code = result.get("code", "")
    if not code:
        raise RuntimeError("Cognito did not return an authorization code.")

    token_url = f"{domain}/oauth2/token"
    payload = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        token_payload = json.loads(response.read().decode("utf-8"))
    return CognitoToken(
        access_token=str(token_payload["access_token"]),
        id_token=str(token_payload.get("id_token", "")),
        expires_in=int(token_payload.get("expires_in", 3600)),
    )
