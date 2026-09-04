from __future__ import annotations

import csv
import json
import os
from io import StringIO
from typing import Any, Dict, List

import boto3

from auth import allowed_site_ids

s3 = boto3.client("s3")

BUCKET = os.environ["DATA_LAKE_BUCKET"]
SITE_IDS_KEY = os.environ.get("SITE_IDS_KEY", "config/site_ids.csv")


def _response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "OPTIONS,GET",
        },
        "body": json.dumps(payload),
    }


def _normalize_site_ids(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().upper()
        if not value or value in {"NAN", "NONE", "<NA>"}:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return sorted(out)


def _parse_site_ids_csv(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames:
        lower_map = {str(name).strip().lower(): name for name in reader.fieldnames}
        col = lower_map.get("site_id") or lower_map.get("site") or lower_map.get("siteid")
        if col:
            return _normalize_site_ids([str(row.get(col, "")) for row in reader])

    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        return []
    if rows[0].strip().lower() in {"site_id", "site", "siteid"}:
        rows = rows[1:]
    return _normalize_site_ids(rows)


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    if event.get("httpMethod") == "OPTIONS":
        return _response(200, {"ok": True})

    try:
        resp = s3.get_object(Bucket=BUCKET, Key=SITE_IDS_KEY)
        raw = (resp.get("Body").read() if resp.get("Body") is not None else b"").decode("utf-8", errors="replace")
        all_site_ids = _parse_site_ids_csv(raw)
        allowed = {value.upper() for value in allowed_site_ids(event)}
        site_ids = [value for value in all_site_ids if value in allowed]
        return _response(
            200,
            {
                "site_ids": site_ids,
                "count": len(site_ids),
                "source": "s3",
                "site_ids_key": SITE_IDS_KEY,
            },
        )
    except Exception as exc:
        return _response(
            500,
            {
                "error": "failed_to_load_site_ids",
                "detail": str(exc),
                "site_ids_key": SITE_IDS_KEY,
            },
        )
