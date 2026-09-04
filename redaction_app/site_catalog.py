from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DEFAULT_SITE_IDS: List[str] = [
    "BCH", "CHA", "CHP", "GOS", "LCH", "MOT", "NCH", "TCH", "TSK", "PIT", "IND", "SCH",
    "ARK", "OUH", "YAL", "MTS", "JHU", "CSH", "SEA", "FCC", "UKY", "COL", "NAT", "VMC",
    "STL", "UOI", "HDC", "CHO", "SAN", "LAX", "WIN", "CMH", "PHX", "DEL", "CHC", "UOC",
    "MON", "NED", "NEF", "MIN", "DAL", "STJ", "BAN", "CIN", "ERA", "SHK", "BER", "OHS", "VCU",
]

DEFAULT_SITE_IDS_BUCKET = "safe-harbor-data-lake"
DEFAULT_SITE_IDS_KEY = "config/site_ids.csv"


def _normalize_site_ids(values: Iterable[object]) -> List[str]:
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
    return out


def _sorted_site_ids(values: Iterable[object]) -> List[str]:
    normalized = _normalize_site_ids(values)
    return sorted(normalized)


def _read_site_ids_csv(csv_text: str) -> List[str]:
    text = str(csv_text or "").strip()
    if not text:
        return []
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        if not rows:
            return []
        if rows[0].lower() in {"site_id", "site", "siteid"}:
            rows = rows[1:]
        return _sorted_site_ids(rows)

    preferred_cols = ["site_id", "site", "siteid", "Site ID", "SiteID"]
    col = next((c for c in preferred_cols if c in df.columns), None)
    if col is None and len(df.columns) > 0:
        col = str(df.columns[0])
    if col is None:
        return []
    return _sorted_site_ids(df[col].tolist())


def _read_site_ids_from_api(api_url: str, api_key: str = "") -> List[str]:
    url = str(api_url or "").strip()
    if not url:
        return []
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    if str(api_key or "").strip():
        req.add_header("x-api-key", str(api_key).strip())
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            site_ids = parsed.get("site_ids", [])
        elif isinstance(parsed, list):
            site_ids = parsed
        else:
            site_ids = []
        return _sorted_site_ids(site_ids)
    except Exception:
        return []


def load_site_ids(
    *,
    local_csv_path: str | None = None,
    bucket: str | None = None,
    key: str | None = None,
) -> List[str]:
    """
    Load Site IDs for dropdown controls.

    Resolution order:
    1) API endpoint (SAFEHARBOR_SITE_IDS_API_URL)
    2) Local override path (arg or SAFEHARBOR_SITE_IDS_CSV)
    3) S3 CSV (bucket/key or env SAFEHARBOR_SITE_IDS_BUCKET/SAFEHARBOR_SITE_IDS_KEY)
    4) Repository default CSV next to this module (site_ids.csv)
    5) Built-in default list
    """
    api_url = str(os.getenv("SAFEHARBOR_SITE_IDS_API_URL", "")).strip()
    api_key = str(os.getenv("SAFEHARBOR_API_KEY", "")).strip()
    if api_url:
        parsed = _read_site_ids_from_api(api_url, api_key=api_key)
        if parsed:
            return parsed

    resolved_local = str(local_csv_path or os.getenv("SAFEHARBOR_SITE_IDS_CSV", "")).strip()
    if resolved_local:
        try:
            text = Path(resolved_local).expanduser().read_text(encoding="utf-8")
            parsed = _read_site_ids_csv(text)
            if parsed:
                return parsed
        except Exception:
            pass

    resolved_bucket = str(bucket or os.getenv("SAFEHARBOR_SITE_IDS_BUCKET", DEFAULT_SITE_IDS_BUCKET)).strip()
    resolved_key = str(key or os.getenv("SAFEHARBOR_SITE_IDS_KEY", DEFAULT_SITE_IDS_KEY)).strip()
    if resolved_bucket and resolved_key:
        try:
            import boto3  # optional runtime dependency

            client = boto3.client("s3")
            obj = client.get_object(Bucket=resolved_bucket, Key=resolved_key)
            body = obj.get("Body")
            payload = body.read() if body is not None else b""
            text = payload.decode("utf-8", errors="replace")
            parsed = _read_site_ids_csv(text)
            if parsed:
                return parsed
        except Exception:
            pass

    packaged_csv = Path(__file__).resolve().with_name("site_ids.csv")
    if packaged_csv.exists():
        try:
            parsed = _read_site_ids_csv(packaged_csv.read_text(encoding="utf-8"))
            if parsed:
                return parsed
        except Exception:
            pass

    return _sorted_site_ids(DEFAULT_SITE_IDS)


if __name__ == "__main__":
    rows = ["site_id", *load_site_ids()]
    out_path = Path(__file__).resolve().with_name("site_ids.csv")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(str(out_path))
