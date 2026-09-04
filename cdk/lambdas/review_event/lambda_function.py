from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import boto3

from auth import resolve_site


ddb = boto3.resource("dynamodb")
table = ddb.Table(os.environ["REVIEW_TABLE"])
site_keys_table = ddb.Table(os.environ.get("SITE_API_KEYS_TABLE", "")) if os.environ.get("SITE_API_KEYS_TABLE") else None


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


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
        return _response(403, {"error": "user is not assigned to the requested site", "detail": site_err})

    file_id = str(body.get("file_id") or body.get("report_id") or body.get("doc_id") or "")
    reviewer = str(body.get("reviewer") or "")
    review_status = str(body.get("review_status") or "")
    review_notes = str(body.get("review_notes") or "")
    approved = int(bool(body.get("approved_to_send", 0)))
    phi_found = _safe_int(body.get("phi_found", 0), 0)
    total_pii_spans = _safe_int(body.get("total_pii_spans", 0), 0)
    force_id = str(body.get("force_id") or "")
    modality_type = str(body.get("modality_type") or "")
    study_date = str(body.get("study_date") or "")
    raw_study_date = str(body.get("raw_study_date") or study_date or "")
    dup = _safe_int(body.get("dup", 0), 0)
    modality_instance = max(1, _safe_int(body.get("modality_instance", 1), 1))
    file_type = str(body.get("file_type") or "")
    documents_in_file = _safe_int(body.get("documents_in_file", 0), 0)
    auto_redaction_elements = _safe_int(body.get("auto_redaction_elements", 0), 0)
    user_deleted_elements = _safe_int(body.get("user_deleted_elements", 0), 0)
    user_added_elements = _safe_int(body.get("user_added_elements", 0), 0)

    if not file_id:
        return _response(400, {"error": "file_id is required"})

    recorded_at = _utc_now_iso()
    item = {
        "pk": f"SITE#{site_id}",
        "sk": f"REVIEW#{file_id}",
        "site_id": site_id,
        "site_acronym": site_acronym,
        "file_id": file_id,
        # compatibility aliases
        "report_id": file_id,
        "doc_id": file_id,
        "recorded_at_utc": recorded_at,
        "event_type": "review_event",
        "reviewer": reviewer,
        "review_status": review_status,
        "review_notes": review_notes,
        "approved_to_send": approved,
        "phi_found": phi_found,
        "total_pii_spans": total_pii_spans,
        "force_id": force_id,
        "modality_type": modality_type,
        "study_date": study_date,
        "raw_study_date": raw_study_date,
        "dup": dup,
        "modality_instance": modality_instance,
        "file_type": file_type,
        "documents_in_file": documents_in_file,
        "auto_redaction_elements": auto_redaction_elements,
        "user_deleted_elements": user_deleted_elements,
        "user_added_elements": user_added_elements,
    }
    table.put_item(Item=item)
    return _response(200, {"ok": True, "saved": item})
