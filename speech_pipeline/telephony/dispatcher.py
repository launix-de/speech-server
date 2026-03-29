"""Event dispatcher — fires subscriber events and processes command responses.

There is no automatic SIP listener.  All calls are created explicitly
by subscribers via ``POST /api/calls``.  SIP legs, webclient slots,
TTS bots etc. are added via commands.

Inbound calls from a SIP trunk arrive as a webhook from Asterisk's
dialplan (``POST /api/calls/inbound``) — see the API blueprint.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import requests as http_requests

from . import call_state, commands, subscriber

_LOGGER = logging.getLogger("telephony.dispatcher")


def _resolve_event_spec(events: dict, event_key: str) -> tuple[str, str] | None:
    event_spec = events.get(event_key) if events else None
    if not event_spec:
        return None

    parts = event_spec.split(None, 1)
    method = "POST"
    path = event_spec
    if len(parts) == 2 and parts[0].upper() in ("GET", "POST", "PUT", "PATCH"):
        method = parts[0].upper()
        path = parts[1]
    return method, path


def fire_subscriber_event(sub: dict, event_key: str, payload: dict) -> list:
    """Send a subscriber-scoped event without requiring an existing call."""
    resolved = _resolve_event_spec(sub.get("events", {}), event_key)
    if not resolved:
        return []
    method, path = resolved

    url = sub["base_url"].rstrip("/") + "/" + path.lstrip("/")
    event_payload = {"event": event_key, **payload}

    def _send() -> None:
        try:
            resp = http_requests.request(
                method,
                url,
                json=event_payload,
                headers={"Authorization": f"Bearer {sub['bearer_token']}"},
                timeout=30,
            )
            _LOGGER.info("Event %s → %d", event_key, resp.status_code)
        except Exception as e:
            _LOGGER.warning("Event %s to %s failed: %s", event_key, url, e)

    threading.Thread(target=_send, daemon=True).start()
    _LOGGER.info("Event %s fired (fire-and-forget) → %s", event_key, url)
    return []


def fire_event(call: call_state.Call, event_key: str,
               payload: dict) -> list:
    """Send an event to the subscriber and return commands from response.

    Looks up the event URL from the call's events map.  If the
    subscriber didn't subscribe to this event, returns [].
    """
    sub = subscriber.get(call.subscriber_id)
    if not sub:
        return []
    return fire_subscriber_event(sub, event_key, {
        "callId": call.call_id,
        **payload,
    })


def fire_event_and_execute(call: call_state.Call, event_key: str,
                           payload: dict) -> None:
    """Fire event, then execute any commands from the response."""
    cmds = fire_event(call, event_key, payload)
    if cmds:
        commands.execute_commands(call, cmds)
