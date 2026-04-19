"""Scoped runtime IDs and local/global name resolution."""
from __future__ import annotations

import secrets
from typing import Optional


def new_local_id(kind: str, entropy_bytes: int = 12) -> str:
    return f"{kind}-{secrets.token_urlsafe(entropy_bytes)}"


def scoped_id(owner: str, kind: str, entropy_bytes: int = 12) -> str:
    """Return ``owner:kind-random``.

    ``owner`` is the authenticated namespace (account / CRM username).
    """
    return f"{owner}:{new_local_id(kind, entropy_bytes=entropy_bytes)}"


def local_id(value: str) -> str:
    return value.split(":", 1)[1] if ":" in value else value


def expand_for_account(value: str, account_id: Optional[str]) -> str:
    if account_id is None:
        return value
    return f"{account_id}:{value}"


def localize_for_account(value: str, account_id: Optional[str]) -> str:
    if account_id is None or not value.startswith(f"{account_id}:"):
        return value
    return value[len(account_id) + 1:]


def localize_fields(payload: dict, account_id: Optional[str], *fields: str) -> dict:
    if account_id is None:
        return dict(payload)
    out = dict(payload)
    for field in fields:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = localize_for_account(value, account_id)
    return out
