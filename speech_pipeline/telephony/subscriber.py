"""Subscriber registry with heartbeat-based liveliness tracking.

Subscribers are ephemeral — held in memory only.  Each subscriber
belongs to an account and refreshes itself via periodic heartbeat
(``PUT /api/subscribe/{id}``).  Stale subscribers are flagged after
``STALE_SECONDS`` and removed after ``REMOVE_SECONDS``.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

_LOGGER = logging.getLogger("telephony.subscriber")

# Events the server actually fires; a subscriber that omits one of
# these will silently drop the corresponding webhook.  Keep in sync
# with sip_listener, dispatcher, pipe_executor.
_EXPECTED_EVENT_KEYS = ("incoming", "call_ended", "device_dial")

STALE_SECONDS = 0   # disabled — subscribers are permanent once registered
REMOVE_SECONDS = 0  # disabled

# subscriber_id -> subscriber dict
_subscribers: Dict[str, dict] = {}

# did -> subscriber_id  (reverse index for inbound routing)
_did_map: Dict[str, str] = {}

# sip_domain -> subscriber_id  (reverse index for SIP registration)
_sip_domain_map: Dict[str, str] = {}


def base_url_to_sip_domain(base_url: str) -> str:
    """Derive SIP domain from a CRM base_url.

    This is intentional API design, not an incidental transformation:
    the SIP identity encodes which CRM owns a device/login, so the speech
    server can route REGISTER/loginAction lookups to exactly one CRM without
    broadcasting credentials or inbound traffic to every tenant.

    Path segments are reversed and prepended as subdomains:
      https://launix.de/crm         -> crm.launix.de
      https://launix.de/fop/crm-neu -> crm-neu.fop.launix.de
      https://crm-neu.launix.de     -> crm-neu.launix.de
    """
    parsed = urlparse(base_url)
    hostname = parsed.hostname or ""
    path = parsed.path.strip("/")
    if not path:
        return hostname
    segments = path.split("/")
    segments.reverse()
    return ".".join(segments) + "." + hostname


def put(subscriber_id: str, account_id: str, data: dict) -> dict:
    """Register or refresh a subscriber (idempotent)."""
    old = _subscribers.get(subscriber_id)
    if old and old["account_id"] != account_id:
        raise PermissionError("Subscriber belongs to a different account")

    # Rebuild DID map entries for this subscriber
    if old:
        for did in old.get("inbound_dids", []):
            _did_map.pop(did, None)

    entry = {
        "id": subscriber_id,
        "account_id": account_id,
        "base_url": data.get("base_url", ""),
        "bearer_token": data.get("bearer_token", ""),
        "outbound_caller_id": data.get("outbound_caller_id", ""),
        "inbound_dids": data.get("inbound_dids", []),
        "events": data.get("events", {}),
        "last_seen": time.time(),
    }

    # Check DID uniqueness
    for did in entry["inbound_dids"]:
        owner = _did_map.get(did)
        if owner and owner != subscriber_id:
            raise ValueError(
                f"DID {did} already claimed by subscriber {owner}")
        _did_map[did] = subscriber_id

    # Update SIP domain reverse index.  This is the authoritative routing map
    # for SIP-client identities -> owning CRM subscriber.  The speech server
    # must never fan out auth or inbound call events to multiple CRMs.
    if old:
        old_domain = base_url_to_sip_domain(old.get("base_url", ""))
        if old_domain:
            _sip_domain_map.pop(old_domain, None)
    sip_domain = base_url_to_sip_domain(entry["base_url"])
    if sip_domain:
        _sip_domain_map[sip_domain] = subscriber_id

    _subscribers[subscriber_id] = entry
    _LOGGER.info("Subscriber %s registered (account=%s, dids=%s, sip_domain=%s)",
                 subscriber_id, account_id, entry["inbound_dids"], sip_domain)

    # Drift guard: server fires a known set of events; if the caller
    # registers without some of them the corresponding webhooks
    # silently vanish.  Warn loudly so CI / pm2 logs surface it.
    missing = [k for k in _EXPECTED_EVENT_KEYS if k not in entry["events"]]
    if missing:
        _LOGGER.warning(
            "Subscriber %s registered without expected events %s — "
            "the server fires those but no callback URL is configured; "
            "refresh heartbeat.fop or check the events dict.",
            subscriber_id, missing,
        )
    return entry


def delete(subscriber_id: str, account_id: Optional[str] = None) -> bool:
    """Remove a subscriber.  If *account_id* is given, ownership is checked."""
    entry = _subscribers.get(subscriber_id)
    if not entry:
        return False
    if account_id and entry["account_id"] != account_id:
        raise PermissionError("Subscriber belongs to a different account")
    for did in entry.get("inbound_dids", []):
        _did_map.pop(did, None)
    sip_domain = base_url_to_sip_domain(entry.get("base_url", ""))
    if sip_domain:
        _sip_domain_map.pop(sip_domain, None)
    del _subscribers[subscriber_id]
    _LOGGER.info("Subscriber %s removed", subscriber_id)
    return True


def get(subscriber_id: str) -> Optional[dict]:
    return _subscribers.get(subscriber_id)


def list_all(account_id: Optional[str] = None) -> List[dict]:
    """List subscribers, optionally filtered by account.

    Automatically purges entries past ``REMOVE_SECONDS`` and annotates
    each entry with a ``status`` field (``alive`` or ``stale``).
    """
    _purge()
    now = time.time()
    result = []
    for s in _subscribers.values():
        if account_id and s["account_id"] != account_id:
            continue
        age = now - s["last_seen"]
        s_copy = dict(s)
        s_copy["status"] = "stale" if age > STALE_SECONDS else "alive"
        result.append(s_copy)
    return result


def find_by_sip_domain(sip_domain: str) -> Optional[dict]:
    """Look up subscriber by SIP domain (derived from base_url)."""
    sid = _sip_domain_map.get(sip_domain)
    if not sid:
        return None
    return _subscribers.get(sid)


def find_by_did(did: str) -> Optional[dict]:
    """Look up subscriber by inbound DID number."""
    sid = _did_map.get(did)
    if not sid:
        return None
    entry = _subscribers.get(sid)
    if not entry:
        _did_map.pop(did, None)
        return None
    if STALE_SECONDS:
        age = time.time() - entry["last_seen"]
        if age > STALE_SECONDS:
            return None  # stale — treat as unreachable
    return entry


def find_by_pbx(pbx_id: str) -> Optional[dict]:
    """Find the first alive subscriber whose account is pinned to this PBX.

    Used as fallback when no DID-specific subscriber is found — subscribers
    with empty DID lists act as wildcard receivers for their PBX.

    The important privacy property is still "route to exactly one CRM":
    this is a single-subscriber fallback for that PBX, not a broadcast to
    every subscriber of that PBX.
    """
    from . import auth as auth_mod
    for entry in _subscribers.values():
        if STALE_SECONDS:
            age = time.time() - entry["last_seen"]
            if age > STALE_SECONDS:
                continue
        if not entry.get("inbound_dids"):  # wildcard: no specific DIDs
            if auth_mod.check_pbx_access(entry["account_id"], pbx_id):
                return entry
    return None


def delete_all_for_account(account_id: str) -> int:
    """Remove all subscribers belonging to an account.  Returns count."""
    to_remove = [sid for sid, s in _subscribers.items()
                 if s["account_id"] == account_id]
    for sid in to_remove:
        delete(sid)
    return len(to_remove)


def _purge() -> None:
    """Remove subscribers past REMOVE_SECONDS (disabled if 0)."""
    if not REMOVE_SECONDS:
        return
    now = time.time()
    expired = [sid for sid, s in _subscribers.items()
               if now - s["last_seen"] > REMOVE_SECONDS]
    for sid in expired:
        delete(sid)
