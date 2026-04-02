"""Shared references and utilities for the telephony subsystem.

Avoids circular imports — modules read from here instead of
importing Flask app context.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import requests as http_requests

_LOGGER = logging.getLogger("telephony.shared")

# Set by piper_multi_server.py during create_app()
tts_registry: Optional[Any] = None  # TTSRegistry
flask_app: Optional[Any] = None


def subscriber_url(sub: dict, path: str) -> str:
    """Build a full URL from a subscriber's base_url and a relative path."""
    return sub["base_url"].rstrip("/") + "/" + path.lstrip("/")


def post_webhook(url: str, payload: dict, bearer_token: str,
                 *, on_response=None) -> None:
    """Fire-and-forget HTTP POST to a subscriber webhook.

    Runs in a daemon thread. Logs result. If *on_response* is given,
    it is called with the Response object on success.
    """
    def _send():
        try:
            resp = http_requests.post(url, json=payload, headers={
                "Authorization": f"Bearer {bearer_token}",
            }, timeout=10)
            _LOGGER.info("Webhook %s → %d", url.split("?")[0].rsplit("/", 1)[-1], resp.status_code)
            if on_response:
                on_response(resp)
        except Exception as e:
            _LOGGER.warning("Webhook %s failed: %s", url, e)

    threading.Thread(target=_send, daemon=True).start()


def ensure_pipe_executor(call) -> Any:
    """Ensure a call has a CallPipeExecutor, creating one if needed."""
    from .pipe_executor import CallPipeExecutor
    from . import subscriber

    if not hasattr(call, 'pipe_executor') or call.pipe_executor is None:
        sub = subscriber.get(call.subscriber_id) if hasattr(call, 'subscriber_id') else None
        call.pipe_executor = CallPipeExecutor(
            call, tts_registry=tts_registry, subscriber=sub)
    return call.pipe_executor
