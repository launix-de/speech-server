"""SIP listener — registers as a SIP client on each PBX.

When the PBX routes an incoming call to us, we create a Leg,
fire the subscriber's ``incoming`` webhook, and wait for the
subscriber to bridge the leg into a conference via the API.

If the subscriber doesn't respond within the timeout, the call
is rejected.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict

from . import leg as leg_mod, subscriber as sub_mod, dispatcher

_LOGGER = logging.getLogger("telephony.sip-listener")

RING_TIMEOUT = 15  # seconds to wait for subscriber to accept

_phones: Dict[str, object] = {}  # pbx_id -> VoIPPhone


def start_listener(pbx_id: str, pbx_entry: dict) -> None:
    """Start a SIP listener for one PBX.  Safe to call multiple times."""
    if pbx_id in _phones:
        return

    try:
        from pyVoIP.VoIP.VoIP import VoIPPhone
    except ImportError:
        _LOGGER.warning("pyVoIP not installed — SIP listener disabled")
        return

    from speech_pipeline.SIPSession import _find_free_udp_port, _patch_voip_phone

    sip_proxy = pbx_entry.get("sip_proxy", "127.0.0.1")
    sip_port = int(pbx_entry.get("sip_port", 5060))
    sip_user = pbx_entry.get("sip_user", "piper")
    sip_pass = pbx_entry.get("sip_password", "")
    local_port = _find_free_udp_port()

    def _on_incoming(voip_call):
        threading.Thread(
            target=_handle_incoming,
            args=(pbx_id, voip_call),
            daemon=True,
        ).start()

    phone = VoIPPhone(
        server=sip_proxy,
        port=sip_port,
        username=sip_user,
        password=sip_pass,
        callCallback=_on_incoming,
        sipPort=local_port,
        myIP=sip_proxy,
    )
    _patch_voip_phone(phone)
    phone.start()
    _phones[pbx_id] = phone
    _LOGGER.info("SIP listener started: %s (%s:%d as %s)",
                 pbx_id, sip_proxy, sip_port, sip_user)


def stop_listener(pbx_id: str) -> None:
    phone = _phones.pop(pbx_id, None)
    if phone:
        try:
            phone.stop()
        except Exception:
            pass


def stop_all() -> None:
    for pbx_id in list(_phones):
        stop_listener(pbx_id)


def _handle_incoming(pbx_id: str, voip_call) -> None:
    """Handle one incoming SIP call."""
    from pyVoIP.VoIP.VoIP import CallState

    # Extract caller/callee
    try:
        req = voip_call.request
        caller = req.headers.get("From", {}).get("number", "unknown")
        callee = req.headers.get("To", {}).get("number", "unknown")
    except Exception:
        caller = "unknown"
        callee = "unknown"

    _LOGGER.info("Incoming SIP on %s: %s → %s", pbx_id, caller, callee)

    # Find subscriber by DID
    sub = sub_mod.find_by_did(callee)
    if not sub:
        _LOGGER.warning("No subscriber for DID %s — rejecting", callee)
        try:
            voip_call.deny()
        except Exception:
            pass
        return

    # Answer the SIP call (we're bridging it ourselves)
    try:
        voip_call.answer()
    except Exception as e:
        _LOGGER.warning("Failed to answer: %s", e)
        return

    # Create an inbound leg
    leg = leg_mod.create_leg(
        direction="inbound",
        number=caller,
        pbx_id=pbx_id,
        subscriber_id=sub["id"],
        voip_call=voip_call,
    )
    leg.status = "ringing"

    # Fire incoming event to subscriber with leg_id
    # Subscriber responds with commands (e.g. create call + add_leg)
    cmds = dispatcher.fire_event(
        # We need a minimal "call-like" object for fire_event
        _EventContext(sub, leg),
        "incoming",
        {"caller": caller, "callee": callee, "leg_id": leg.leg_id},
    )

    if cmds:
        # Execute commands — they should bridge this leg into a conference
        from . import commands as cmd_engine, call_state
        # Commands need a call context; if subscriber created one, find it
        # For now, execute in a temporary context
        _execute_inbound_commands(leg, sub, cmds)
    else:
        # No commands — wait for subscriber to call the bridge API
        _wait_for_bridge(leg, voip_call)


def _execute_inbound_commands(leg, sub: dict, cmds: list) -> None:
    """Execute commands that came from the incoming event response."""
    from . import commands as cmd_engine, call_state

    for cmd in cmds:
        action = cmd.get("action")
        if action == "create_call":
            # Create a conference and bridge this leg in
            call = call_state.create_call(
                subscriber_id=sub["id"],
                account_id=sub["account_id"],
                pbx_id=leg.pbx_id,
                caller=leg.number,
                direction="inbound",
                events=sub.get("events", {}),
            )
            callbacks = cmd.get("callbacks", {})
            leg.callbacks.update(callbacks)
            leg_mod.bridge_to_call(leg, call)

            # Execute remaining commands on this call
            remaining = [c for c in cmds if c is not cmd]
            if remaining:
                cmd_engine.execute_commands(call, remaining)
            return

        elif action == "add_leg" and cmd.get("leg_id") == leg.leg_id:
            call_id = cmd.get("call_id")
            if call_id:
                call = call_state.get_call(call_id)
                if call:
                    leg_mod.bridge_to_call(leg, call)
                    return

    # If no bridge command, wait
    _wait_for_bridge(leg, leg.voip_call)


def _wait_for_bridge(leg, voip_call) -> None:
    """Wait for subscriber to bridge the leg via API, or timeout."""
    from pyVoIP.VoIP.VoIP import CallState

    deadline = time.time() + RING_TIMEOUT
    while time.time() < deadline:
        if leg.call_id:
            return  # bridged via API
        if voip_call.state == CallState.ENDED:
            leg.status = "completed"
            leg_mod.delete_leg(leg.leg_id)
            return
        time.sleep(0.5)

    # Timeout — reject
    _LOGGER.warning("Leg %s: no bridge within %ds — hanging up",
                    leg.leg_id, RING_TIMEOUT)
    leg.status = "no-answer"
    try:
        voip_call.hangup()
    except Exception:
        pass
    leg_mod.delete_leg(leg.leg_id)


class _EventContext:
    """Minimal object that looks like a Call for fire_event()."""
    def __init__(self, sub: dict, leg):
        self.call_id = leg.leg_id  # use leg_id as event context
        self.subscriber_id = sub["id"]
        self.events = sub.get("events", {})
