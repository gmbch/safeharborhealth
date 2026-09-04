from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import boto3

from auth import resolve_site


ddb = boto3.resource("dynamodb")
site_keys_table = ddb.Table(os.environ["SITE_API_KEYS_TABLE"])
session_table = ddb.Table(os.environ["SESSION_TABLE"])


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


def _api_key_id_from_event(event: Dict[str, Any]) -> str:
    return str((((event.get("requestContext") or {}).get("identity") or {}).get("apiKeyId")) or "")


def _resolve_site(api_key_id: str) -> tuple[str, str]:
    item = (site_keys_table.get_item(Key={"api_key_id": api_key_id}) or {}).get("Item") or {}
    if not item:
        return "", ""
    if str(item.get("status") or "active").lower() != "active":
        return "", ""
    return str(item.get("site_id") or "").strip(), str(item.get("site_acronym") or "").strip()


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    if event.get("httpMethod") == "OPTIONS":
        return _response(200, {"ok": True})

    body = json.loads(event.get("body") or "{}")
    site_id, site_error = resolve_site(event, str(body.get("site_id") or ""))
    site_acronym = site_id
    if not site_id:
        return _response(403, {"error": site_error or "user is not assigned to a site"})
    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        return _response(400, {"error": "session_id is required"})

    key = {"pk": f"SITE#{site_id}", "sk": f"SESSION#{session_id}"}
    existing = (session_table.get_item(Key=key) or {}).get("Item") or {}
    if not existing:
        return _response(404, {"error": "session_id not found for site", "session_id": session_id, "site_id": site_id})

    now = _utc_now_iso()
    session_table.update_item(
        Key=key,
        UpdateExpression="SET #status = :status, close_requested_at_utc = :closed, status_updated_utc = :updated",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={
            ":status": "close_requested",
            ":closed": now,
            ":updated": now,
        },
    )

    return _response(
        200,
        {
            "ok": True,
            "site_id": site_id,
            "site_acronym": site_acronym,
            "session_id": session_id,
            "status": "close_requested",
        },
    )
