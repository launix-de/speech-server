"""Two-tier authentication: admin token + per-account tokens.

The admin token is set once at startup via ``init()``.  Account tokens
are created dynamically through the admin API.

Decorators
----------
- ``require_admin``  – only the ``--admin-token`` is accepted.
- ``require_account`` – any valid account token (or admin) is accepted.
  The resolved *account* dict is stored in ``g.account``; admin requests
  get ``g.account = None``.
"""
from __future__ import annotations

import functools
import logging
import secrets
import time
from typing import Dict, Optional

from flask import g, request

_LOGGER = logging.getLogger("telephony.auth")

_admin_token: Optional[str] = None

# account_id -> {id, token, pbx, max_concurrent_calls, features}
_accounts: Dict[str, dict] = {}

# nonce -> {nonce, subscriber_id, user, account_id, created, ttl, ws?}
_nonces: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init(admin_token: str) -> None:
    """Set the admin bearer token (called once at startup)."""
    global _admin_token
    _admin_token = admin_token


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def _extract_token() -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def require_admin(f):
    """Reject requests that do not carry the admin token."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not _admin_token:
            return ("Telephony API disabled (no --admin-token)\n", 403)
        token = _extract_token()
        if token != _admin_token:
            return ("Forbidden\n", 403)
        g.account = None
        return f(*args, **kwargs)
    return wrapper


def require_account(f):
    """Accept admin token *or* any valid account token.

    On success ``flask.g.account`` is set to the account dict (or
    ``None`` when the admin token was used).
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not _admin_token:
            return ("Telephony API disabled (no --admin-token)\n", 403)
        token = _extract_token()
        if not token:
            return ("Missing Authorization: Bearer <token>\n", 401)
        if token == _admin_token:
            g.account = None          # admin
            return f(*args, **kwargs)
        acct = _find_account_by_token(token)
        if not acct:
            return ("Forbidden\n", 403)
        g.account = acct
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Account CRUD (called from telephony_api blueprint)
# ---------------------------------------------------------------------------

def put_account(account_id: str, data: dict) -> dict:
    token = data.get("token")
    if not token:
        raise ValueError("'token' is required")
    acct = {
        "id": account_id,
        "token": token,
        "pbx": data.get("pbx"),             # None = any PBX allowed
        "max_concurrent_calls": data.get("max_concurrent_calls", 0),
        "features": data.get("features", []),
    }
    _accounts[account_id] = acct
    _LOGGER.info("Account registered: %s (pbx=%s)", account_id, acct["pbx"])
    return acct


def delete_account(account_id: str) -> bool:
    if account_id in _accounts:
        del _accounts[account_id]
        _LOGGER.info("Account removed: %s", account_id)
        return True
    return False


def get_account(account_id: str) -> Optional[dict]:
    return _accounts.get(account_id)


def list_accounts() -> list:
    return list(_accounts.values())


def _find_account_by_token(token: str) -> Optional[dict]:
    for acct in _accounts.values():
        if acct["token"] == token:
            return acct
    return None


# ---------------------------------------------------------------------------
# Nonce management
# ---------------------------------------------------------------------------

def create_nonce(account_id: str, subscriber_id: str, user: str,
                 ttl: int = 3600) -> dict:
    nonce_val = "n-" + secrets.token_urlsafe(24)
    entry = {
        "nonce": nonce_val,
        "account_id": account_id,
        "subscriber_id": subscriber_id,
        "user": user,
        "created": time.time(),
        "ttl": ttl,
        "connected": False,
    }
    _nonces[nonce_val] = entry
    return entry


def validate_nonce(nonce: str) -> Optional[dict]:
    entry = _nonces.get(nonce)
    if not entry:
        return None
    if time.time() - entry["created"] > entry["ttl"]:
        del _nonces[nonce]
        return None
    return entry


def revoke_nonce(nonce: str) -> bool:
    if nonce in _nonces:
        del _nonces[nonce]
        return True
    return False


def list_nonces(account_id: Optional[str] = None) -> list:
    now = time.time()
    result = []
    expired = []
    for k, v in _nonces.items():
        if now - v["created"] > v["ttl"]:
            expired.append(k)
            continue
        if account_id and v["account_id"] != account_id:
            continue
        result.append(v)
    for k in expired:
        del _nonces[k]
    return result
