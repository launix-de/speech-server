"""PBX registry — tracks Asterisk (or other SIP) instances.

Each PBX entry stores connection details needed to originate and manage
SIP calls.  Entries are held in memory only and re-provisioned via the
startup callback on restart.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

_LOGGER = logging.getLogger("telephony.pbx")

# pbx_id -> pbx dict
_pbx_list: Dict[str, dict] = {}


def put(pbx_id: str, data: dict) -> dict:
    """Register or update a PBX."""
    entry = {
        "id": pbx_id,
        "sip_proxy": data.get("sip_proxy", ""),
        "ari_url": data.get("ari_url", ""),
        "ari_user": data.get("ari_user", ""),
        "ari_password": data.get("ari_password", ""),
    }
    _pbx_list[pbx_id] = entry
    _LOGGER.info("PBX registered: %s (proxy=%s)", pbx_id, entry["sip_proxy"])
    return _public(entry)


def delete(pbx_id: str) -> bool:
    if pbx_id in _pbx_list:
        del _pbx_list[pbx_id]
        _LOGGER.info("PBX removed: %s", pbx_id)
        return True
    return False


def get(pbx_id: str) -> Optional[dict]:
    entry = _pbx_list.get(pbx_id)
    return dict(entry) if entry else None


def list_all() -> List[dict]:
    return [_public(p) for p in _pbx_list.values()]


def _public(entry: dict) -> dict:
    """Return a copy without sensitive fields (passwords)."""
    return {k: v for k, v in entry.items() if k != "ari_password"}
