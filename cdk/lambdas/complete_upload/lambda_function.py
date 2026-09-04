from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from auth import resolve_site


s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")
sqs = boto3.client("sqs")

BUCKET = os.environ["DATA_LAKE_BUCKET"]
TABLE = os.environ["REVIEW_TABLE"]
TASKS_TABLE = os.environ.get("REVIEW_TASKS_TABLE", "")
SITE_API_KEYS_TABLE = os.environ.get("SITE_API_KEYS_TABLE", "")
EXTRACTION_QUEUE_URL = os.environ.get("EXTRACTION_QUEUE_URL", "")
SESSION_TABLE = os.environ.get("SESSION_TABLE", "")
table = ddb.Table(TABLE)
tasks_table = ddb.Table(TASKS_TABLE) if TASKS_TABLE else None
site_keys_table = ddb.Table(SITE_API_KEYS_TABLE) if SITE_API_KEYS_TABLE else None
session_table = ddb.Table(SESSION_TABLE) if SESSION_TABLE else None
logger = logging.getLogger("safeharbor-complete-upload")
logger.setLevel(logging.INFO)


def _log(event_type: str, **kwargs: Any) -> None:
    logger.info(json.dumps({"event_type": event_type, **kwargs}))


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


def _resolve_session_queue_url(site_id: str, session_id: str) -> str:
    if not (session_id and session_table is not None):
        return ""
    key = {"pk": f"SITE#{site_id}", "sk": f"SESSION#{session_id}"}
    item = (session_table.get_item(Key=key) or {}).get("Item") or {}
    return str(item.get("session_queue_url") or "").strip()


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    if event.get("httpMethod") == "OPTIONS":
        return _response(200, {"ok": True})

    body = json.loads(event.get("body") or "{}")
    site_id, site_acronym, site_err = _resolve_site_for_request(event, body)
    if not site_id:
        _log("upload_complete_denied", reason=site_err)
        return _response(403, {"error": "user is not assigned to an active site", "detail": site_err})

    file_id = str(body.get("file_id") or body.get("report_id") or body.get("doc_id") or "")
    object_key = str(body.get("object_key") or "")
    reviewer = str(body.get("reviewer") or "")
    session_id = str(body.get("session_id") or "")
    modality_type = str(body.get("modality_type") or "")
    file_type = str(
        body.get("file_type")
        or ("json" if object_key.lower().endswith((".json", ".jsonl")) else "pdf")
    ).strip().lower()
    force_id = str(body.get("force_id") or "")
    study_date = str(body.get("study_date") or "")
    raw_study_date = str(body.get("raw_study_date") or study_date or "")
    dup = _safe_int(body.get("dup", 0), 0)
    modality_instance = max(1, _safe_int(body.get("modality_instance", 1), 1))
    target_queue_url = _resolve_session_queue_url(site_id, session_id) if session_id else ""

    if not file_id or not object_key:
        return _response(400, {"error": "file_id and object_key are required"})
    expected_prefix = f"{os.environ.get('UPLOADS_PREFIX', 'phi-redaction-uploads')}/site_id={site_id}/"
    if not object_key.startswith(expected_prefix):
        return _response(403, {"error": "object_key is outside the authenticated site prefix"})
    if session_id and not target_queue_url:
        return _response(
            400,
            {
                "error": "session_id was provided but no active session queue was found",
                "session_id": session_id,
            },
        )

    try:
        head = s3.head_object(Bucket=BUCKET, Key=object_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        _log("upload_complete_s3_not_found", site_id=site_id, file_id=file_id, object_key=object_key, error_code=code)
        return _response(404, {"error": f"S3 object not found ({code})"})

    recorded_at = _utc_now_iso()
    item = {
        "pk": f"SITE#{site_id}",
        "sk": f"UPLOAD#{file_id}",
        "file_id": file_id,
        # compatibility aliases
        "report_id": file_id,
        "doc_id": file_id,
        "recorded_at_utc": recorded_at,
        "event_type": "upload_completed",
        "bucket": BUCKET,
        "object_key": object_key,
        "content_length": int(head.get("ContentLength", 0)),
        "etag": str(head.get("ETag", "")).strip('"'),
        "reviewer": reviewer,
        "site_acronym": site_acronym,
        "session_id": session_id,
        "modality_type": modality_type,
        "file_type": file_type,
        "force_id": force_id,
        "study_date": study_date,
        "raw_study_date": raw_study_date,
        "dup": dup,
        "modality_instance": modality_instance,
    }
    is_duplicate = False
    try:
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            is_duplicate = True
            _log(
                "upload_complete_duplicate",
                site_id=site_id,
                file_id=file_id,
                object_key=object_key,
                session_id=session_id,
                file_type=file_type,
            )
        else:
            raise

    if not is_duplicate:
        lifecycle_row = {
            "pk": f"SITE#{site_id}",
            "sk": f"TASK#{file_id}",
            "site_id": site_id,
            "site_acronym": site_acronym,
            "file_id": file_id,
            "session_id": session_id,
            "status": "uploaded",
            "status_updated_utc": recorded_at,
            "object_key": object_key,
            "bucket": BUCKET,
            "file_type": file_type,
            "modality_type": modality_type,
            "etag": str(head.get("ETag", "")).strip('"'),
            "force_id": force_id,
            "study_date": study_date,
            "raw_study_date": raw_study_date,
            "dup": dup,
            "modality_instance": modality_instance,
        }
        if tasks_table is not None:
            tasks_table.put_item(Item=lifecycle_row)

    queue_payload = {
        "job_id": uuid.uuid4().hex,
        "site_id": site_id,
        "site_acronym": site_acronym,
        "file_id": file_id,
        "session_id": session_id,
        "bucket": BUCKET,
        "object_key": object_key,
        "file_type": file_type,
        "modality_type": modality_type,
        "enqueued_at_utc": recorded_at,
        "force_id": force_id,
        "study_date": study_date,
        "raw_study_date": raw_study_date,
        "dup": dup,
        "modality_instance": modality_instance,
    }
    if not target_queue_url:
        target_queue_url = EXTRACTION_QUEUE_URL

    should_enqueue_extraction = file_type.strip().lower() == "json"
    if target_queue_url and should_enqueue_extraction:
        session_group = session_id or f"{site_id}-default-session"
        etag = str(head.get("ETag", "")).strip('"')
        dedup_suffix = uuid.uuid4().hex if is_duplicate else etag
        sqs.send_message(
            QueueUrl=target_queue_url,
            MessageBody=json.dumps(queue_payload),
            MessageGroupId=session_group,
            MessageDeduplicationId=f"{site_id}:{file_id}:{dedup_suffix}",
            MessageAttributes={
                "site_id": {"DataType": "String", "StringValue": site_id},
                "session_id": {"DataType": "String", "StringValue": session_id or "none"},
            },
        )
        if tasks_table is not None:
            tasks_table.update_item(
                Key={"pk": f"SITE#{site_id}", "sk": f"TASK#{file_id}"},
                UpdateExpression="SET #status = :status, status_updated_utc = :updated",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={":status": "extraction_queued", ":updated": _utc_now_iso()},
            )
        _log(
            "upload_complete_queued",
            site_id=site_id,
            file_id=file_id,
            session_id=session_id,
            queue_url=target_queue_url,
            object_key=object_key,
            file_type=file_type,
            modality_type=modality_type,
            duplicate=is_duplicate,
        )
    elif target_queue_url and not should_enqueue_extraction:
        _log(
            "upload_complete_skip_enqueue_non_json",
            site_id=site_id,
            file_id=file_id,
            session_id=session_id,
            queue_url=target_queue_url,
            object_key=object_key,
            file_type=file_type,
            modality_type=modality_type,
        )
    else:
        _log(
            "upload_complete_no_queue",
            site_id=site_id,
            file_id=file_id,
            session_id=session_id,
            object_key=object_key,
        )

    return _response(
        200,
        {
            "ok": True,
            "saved": item,
            "duplicate": is_duplicate,
            "queued": bool(target_queue_url and should_enqueue_extraction),
            "queue_url": target_queue_url,
            "queued_reason": "json_extraction_only" if target_queue_url else "no_queue_available",
        },
    )
