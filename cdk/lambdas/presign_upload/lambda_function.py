from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import boto3

from auth import resolve_site


s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
BUCKET = os.environ["DATA_LAKE_BUCKET"]
PREFIX = os.environ.get("UPLOADS_PREFIX", "phi-redaction-uploads")
SITE_API_KEYS_TABLE = os.environ.get("SITE_API_KEYS_TABLE", "")
site_keys_table = dynamodb.Table(SITE_API_KEYS_TABLE) if SITE_API_KEYS_TABLE else None


def _response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps(payload),
    }


def _sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return clean or f"upload-{uuid.uuid4().hex}.pdf"

def _sanitize_modality(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return "unknown"
    clean = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    aliases = {
        "cmr": "cmr",
        "cardiac_mri": "cmr",
        "mri": "cmr",
        "ct": "ct",
        "echo": "echo",
        "echocardiogram": "echo",
        "stresstest": "stress_test",
        "stress_test": "stress_test",
        "stress": "stress_test",
        "cath": "cath",
        "catheterization": "cath",
    }
    return aliases.get(clean, clean or "unknown")


def _norm(value: str) -> str:
    return str(value or "").strip().lower()


def _resolve_site_for_request(event: Dict[str, Any], body: Dict[str, Any]) -> tuple[str, str, str]:
    requested_site_id = str(body.get("site_id") or "").strip()
    requested_site_acronym = str(body.get("site_acronym") or "").strip()
    site_id, error = resolve_site(event, requested_site_id)
    if not site_id:
        return "", "", error
    if requested_site_acronym and _norm(requested_site_acronym) != _norm(site_id):
        return "", "", "site_acronym must match the authenticated site"
    return site_id, site_id, ""


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    if event.get("httpMethod") == "OPTIONS":
        return _response(200, {"ok": True})

    body = json.loads(event.get("body") or "{}")
    site_id, site_acronym, site_err = _resolve_site_for_request(event, body)
    if not site_id:
        return _response(403, {"error": "user is not assigned to an active site", "detail": site_err})

    file_id = str(body.get("file_id") or body.get("report_id") or uuid.uuid4().hex)
    filename = _sanitize_filename(str(body.get("filename") or f"{file_id}.pdf"))
    modality = _sanitize_modality(str(body.get("modality_type") or body.get("modality") or ""))
    content_type = str(body.get("content_type") or "application/pdf")
    expires_seconds = int(body.get("expires_seconds") or 3600)

    object_key = (
        f"{PREFIX}/site_id={site_id}/site_acronym={_sanitize_filename(site_acronym or 'unknown')}"
        f"/modality={modality}/file_id={file_id}/"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{filename}"
    )

    upload_url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": BUCKET,
            "Key": object_key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_seconds,
    )

    return _response(
        200,
        {
            "bucket": BUCKET,
            "object_key": object_key,
            "upload_url": upload_url,
            "expires_seconds": expires_seconds,
            "required_headers": {"Content-Type": content_type},
            "site_id": site_id,
            "site_acronym": site_acronym,
        },
    )
