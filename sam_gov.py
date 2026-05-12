"""SAM.gov Opportunities API integration for RFP fetching."""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta
from typing import Optional

import requests

SAM_API_BASE = "https://api.sam.gov"

# SAM.gov notice types relevant to RFPs
NOTICE_TYPES = {
    "o": "Solicitation",
    "k": "Combined Synopsis/Solicitation",
    "p": "Presolicitation",
    "r": "Sources Sought",
    "s": "Special Notice",
}


def _get_api_key() -> str:
    key = os.environ.get("SAM_GOV_API_KEY", "").strip()
    if not key:
        sys.exit(
            "Error: SAM_GOV_API_KEY environment variable is not set.\n"
            "Get a free API key at https://sam.gov/profile/details and add it to .env"
        )
    return key


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    for entity, char in [
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"'), ("&apos;", "'"),
    ]:
        text = text.replace(entity, char)
    return re.sub(r" {2,}", " ", text).strip()


def _default_date_range(days_back: int = 90) -> tuple[str, str]:
    """Return (postedFrom, postedTo) strings in MM/dd/yyyy format."""
    today = datetime.utcnow()
    posted_to = today.strftime("%m/%d/%Y")
    posted_from = (today - timedelta(days=days_back)).strftime("%m/%d/%Y")
    return posted_from, posted_to


def search_opportunities(
    query: str,
    limit: int = 5,
    posted_from: Optional[str] = None,
    posted_to: Optional[str] = None,
    notice_types: str = "o,k",
) -> list[dict]:
    """
    Search SAM.gov for active RFP opportunities.

    Args:
        query: Keyword search string.
        limit: Number of results to return (max 1000).
        posted_from: Start of date range, MM/dd/yyyy (default: 90 days ago).
        posted_to: End of date range, MM/dd/yyyy (default: today).
        notice_types: Comma-separated SAM.gov ptype codes (default: solicitations + combined).

    Returns:
        List of opportunity dicts from the SAM.gov API.
    """
    default_from, default_to = _default_date_range()
    params: dict = {
        "api_key": _get_api_key(),
        "q": query,
        "limit": limit,
        "ptype": notice_types,
        "postedFrom": posted_from or default_from,
        "postedTo": posted_to or default_to,
    }

    print(
        f"[*] Searching SAM.gov for: {query!r} "
        f"(limit={limit}, {params['postedFrom']} → {params['postedTo']})",
        file=sys.stderr,
    )
    resp = requests.get(
        f"{SAM_API_BASE}/opportunities/v2/search",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("opportunitiesData", [])


def fetch_opportunity_by_id(notice_id: str) -> Optional[dict]:
    """Fetch a single opportunity by its SAM.gov notice ID."""
    params = {"api_key": _get_api_key(), "noticeid": notice_id}
    print(f"[*] Fetching SAM.gov notice: {notice_id}", file=sys.stderr)
    resp = requests.get(
        f"{SAM_API_BASE}/opportunities/v2/search",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    opps = data.get("opportunitiesData", [])
    return opps[0] if opps else None


def opportunity_to_text(notice: dict) -> str:
    """Convert a SAM.gov opportunity dict to a plain-text RFP document."""
    parts: list[str] = ["REQUEST FOR PROPOSAL\n"]

    def _add(label: str, key: str) -> None:
        val = notice.get(key)
        if val:
            parts.append(f"{label}: {val}")

    _add("TITLE", "title")
    _add("SOLICITATION NUMBER", "solicitationNumber")
    _add("ISSUING ORGANIZATION", "fullParentPathName")
    _add("DEPARTMENT/AGENCY", "departmentName")
    _add("SUBTIER", "subtierName")
    _add("OFFICE", "officeName")
    _add("NOTICE TYPE", "type")
    _add("POSTED DATE", "postedDate")
    _add("RESPONSE DEADLINE", "responseDeadLine")
    _add("ARCHIVE DATE", "archiveDate")
    _add("NAICS CODE", "naicsCode")
    _add("NAICS DESCRIPTION", "naicsDescription")
    _add("CLASSIFICATION CODE", "classificationCode")
    _add("SET-ASIDE TYPE", "typeOfSetAsideDescription")
    _add("PLACE OF PERFORMANCE", "placeOfPerformance")

    # Point of contact
    poc = notice.get("pointOfContact")
    if poc and isinstance(poc, list) and poc:
        p = poc[0]
        parts.append("\nPOINT OF CONTACT:")
        if p.get("fullName"):
            parts.append(f"  Name: {p['fullName']}")
        if p.get("email"):
            parts.append(f"  Email: {p['email']}")
        if p.get("phone"):
            parts.append(f"  Phone: {p['phone']}")

    # Description / scope
    desc = notice.get("description") or notice.get("synopsis", "")
    if desc:
        parts.append("\nDESCRIPTION / SCOPE OF WORK:")
        parts.append(_strip_html(desc))

    if notice.get("uiLink"):
        parts.append(f"\nSAM.GOV LINK: {notice['uiLink']}")

    return "\n".join(parts)


def list_attachments(notice_id: str) -> list[dict]:
    """Return metadata for file attachments on a SAM.gov notice."""
    api_key = _get_api_key()
    url = f"{SAM_API_BASE}/opportunities/v1/noticeid/{notice_id}/resources"
    resp = requests.get(url, params={"api_key": api_key}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("resources", data.get("attachments", []))


def download_attachment(notice_id: str, resource_id: str) -> bytes:
    """Download a specific attachment from a SAM.gov notice."""
    api_key = _get_api_key()
    url = (
        f"{SAM_API_BASE}/opportunities/v1/noticeid/{notice_id}"
        f"/resources/files/{resource_id}/download"
    )
    resp = requests.get(url, params={"api_key": api_key}, timeout=60)
    resp.raise_for_status()
    return resp.content
