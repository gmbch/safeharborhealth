from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import boto3

from auth import resolve_site


ddb = boto3.resource("dynamodb")
sqs = boto3.client("sqs")
ecs = boto3.client("ecs")

SITE_API_KEYS_TABLE = os.environ["SITE_API_KEYS_TABLE"]
SESSION_TABLE = os.environ["SESSION_TABLE"]
ECS_CLUSTER_NAME = os.environ["ECS_CLUSTER_NAME"]
ECS_TASK_DEF_ARN = os.environ["ECS_TASK_DEF_ARN"]
ECS_CONTAINER_NAME = os.environ.get("ECS_CONTAINER_NAME", "Worker")
ECS_SUBNET_IDS = [x.strip() for x in str(os.environ.get("ECS_SUBNET_IDS", "")).split(",") if x.strip()]
ECS_SECURITY_GROUP_IDS = [x.strip() for x in str(os.environ.get("ECS_SECURITY_GROUP_IDS", "")).split(",") if x.strip()]
SESSION_IDLE_MAX_POLLS = str(os.environ.get("SESSION_IDLE_MAX_POLLS", "9"))
ENV = os.environ.get("ENV", "dev")

site_keys_table = ddb.Table(SITE_API_KEYS_TABLE)
session_table = ddb.Table(SESSION_TABLE)
logger = logging.getLogger("safeharbor-session-start")
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


def _api_key_id_from_event(event: Dict[str, Any]) -> str:
    return str((((event.get("requestContext") or {}).get("identity") or {}).get("apiKeyId")) or "")


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.strip().lower()).strip("-")


def _resolve_site(api_key_id: str) -> tuple[str, str]:
    item = (site_keys_table.get_item(Key={"api_key_id": api_key_id}) or {}).get("Item") or {}
    if not item:
        return "", ""
    if str(item.get("status") or "active").lower() != "active":
        return "", ""
    return str(item.get("site_id") or "").strip(), str(item.get("site_acronym") or "").strip()


def _create_session_queue(site_id: str, session_id: str) -> tuple[str, str]:
    q_name = f"safeharbor-s-{_sanitize_token(site_id)}-{_sanitize_token(session_id)}.fifo"
    # Keep queue name within SQS 80-char limit.
    q_name = q_name[:75] + ".fifo" if len(q_name) > 80 else q_name
    create = sqs.create_queue(
        QueueName=q_name,
        Attributes={
            "FifoQueue": "true",
            "ContentBasedDeduplication": "true",
            "ReceiveMessageWaitTimeSeconds": "20",
            "VisibilityTimeout": "900",
            "MessageRetentionPeriod": str(4 * 24 * 60 * 60),
            "SqsManagedSseEnabled": "true",
        },
    )
    queue_url = str(create.get("QueueUrl") or "")
    attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"]).get("Attributes", {})
    queue_arn = str(attrs.get("QueueArn") or "")
    return queue_url, queue_arn


def _run_ecs_task(site_id: str, session_id: str, session_queue_url: str) -> str:
    if not ECS_SUBNET_IDS or not ECS_SECURITY_GROUP_IDS:
        raise RuntimeError("ECS subnet/security group configuration is missing")

    resp = ecs.run_task(
        cluster=ECS_CLUSTER_NAME,
        taskDefinition=ECS_TASK_DEF_ARN,
        launchType="FARGATE",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNET_IDS,
                "securityGroups": ECS_SECURITY_GROUP_IDS,
                "assignPublicIp": "DISABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": ECS_CONTAINER_NAME,
                    "environment": [
                        {"name": "EXTRACTION_QUEUE_URL", "value": session_queue_url},
                        {"name": "SESSION_IDLE_MAX_POLLS", "value": SESSION_IDLE_MAX_POLLS},
                        {"name": "SESSION_ID", "value": session_id},
                        {"name": "SESSION_SITE_ID", "value": site_id},
                    ],
                }
            ]
        },
    )
    failures = resp.get("failures", [])
    if failures:
        raise RuntimeError(f"ecs.run_task failures: {failures}")
    tasks = resp.get("tasks", [])
    if not tasks:
        raise RuntimeError("ecs.run_task returned no task")
    return str(tasks[0].get("taskArn") or "")


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    if event.get("httpMethod") == "OPTIONS":
        return _response(200, {"ok": True})

    body = json.loads(event.get("body") or "{}")
    site_id, site_error = resolve_site(event, str(body.get("site_id") or ""))
    site_acronym = site_id
    if not site_id:
        _log("session_start_denied", reason=site_error or "user_not_assigned_to_site")
        return _response(403, {"error": site_error or "user is not assigned to a site"})
    session_id = str(body.get("session_id") or uuid.uuid4().hex[:16]).strip()
    session_label = str(body.get("session_label") or "")
    now = _utc_now_iso()

    existing = (
        session_table.get_item(Key={"pk": f"SITE#{site_id}", "sk": f"SESSION#{session_id}"}) or {}
    ).get("Item")
    if existing:
        _log(
            "session_start_reused",
            site_id=site_id,
            session_id=session_id,
            ecs_task_arn=existing.get("ecs_task_arn", ""),
            session_queue_url=existing.get("session_queue_url", ""),
        )
        return _response(
            200,
            {
                "ok": True,
                "site_id": site_id,
                "site_acronym": site_acronym,
                "session_id": session_id,
                "session_queue_url": existing.get("session_queue_url", ""),
                "ecs_task_arn": existing.get("ecs_task_arn", ""),
                "status": existing.get("status", "started"),
                "reused": True,
            },
        )

    session_queue_url, session_queue_arn = _create_session_queue(site_id, session_id)
    ecs_task_arn = _run_ecs_task(site_id, session_id, session_queue_url)
    item = {
        "pk": f"SITE#{site_id}",
        "sk": f"SESSION#{session_id}",
        "site_id": site_id,
        "site_acronym": site_acronym,
        "session_id": session_id,
        "session_label": session_label,
        "session_queue_url": session_queue_url,
        "session_queue_arn": session_queue_arn,
        "ecs_task_arn": ecs_task_arn,
        "status": "started",
        "started_at_utc": now,
        "env": ENV,
    }
    session_table.put_item(Item=item)
    _log(
        "session_started",
        site_id=site_id,
        site_acronym=site_acronym,
        session_id=session_id,
        ecs_task_arn=ecs_task_arn,
        session_queue_url=session_queue_url,
        session_queue_arn=session_queue_arn,
    )
    return _response(
        200,
        {
            "ok": True,
            "site_id": site_id,
            "site_acronym": site_acronym,
            "session_id": session_id,
            "session_queue_url": session_queue_url,
            "ecs_task_arn": ecs_task_arn,
            "status": "started",
        },
    )
