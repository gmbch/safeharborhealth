from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Set


SITE_GROUP_PREFIX = "safeharbor-site-"


def _claims(event: Dict[str, Any]) -> Dict[str, Any]:
    authorizer = ((event.get("requestContext") or {}).get("authorizer") or {})
    claims = authorizer.get("claims") or {}
    return claims if isinstance(claims, dict) else {}


def username(event: Dict[str, Any]) -> str:
    claims = _claims(event)
    return str(claims.get("email") or claims.get("cognito:username") or claims.get("sub") or "").strip()


def allowed_site_ids(event: Dict[str, Any]) -> Set[str]:
    raw_groups = _claims(event).get("cognito:groups", "")
    groups: Iterable[str]
    if isinstance(raw_groups, str):
        groups = re.split(r"[,\s]+", raw_groups)
    elif isinstance(raw_groups, list):
        groups = (str(item) for item in raw_groups)
    else:
        groups = ()
    out: Set[str] = set()
    for group in groups:
        value = str(group or "").strip().lower()
        if value.startswith(SITE_GROUP_PREFIX):
            site = value[len(SITE_GROUP_PREFIX):].strip()
            if site:
                out.add(site)
    return out


def resolve_site(event: Dict[str, Any], requested_site: str = "") -> tuple[str, str]:
    allowed = allowed_site_ids(event)
    requested = str(requested_site or "").strip().lower()
    if requested and requested not in allowed:
        return "", "requested site is not assigned to this user"
    if not requested:
        if len(allowed) != 1:
            return "", "site_id is required when the user has multiple site assignments"
        requested = next(iter(allowed))
    return requested.upper(), ""
