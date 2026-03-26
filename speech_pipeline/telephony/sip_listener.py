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

RING_TIMEOUT = 30  # seconds to wait for subscriber to bridge via API

_phones: Dict[str, object] = {}  # pbx_id -> VoIPPhone
_INBOUND_DIALOG_TTL = 300.0
_inbound_dialogs: Dict[tuple[str, str], float] = {}
_inbound_dialogs_lock = threading.Lock()


def _claim_inbound_dialog(pbx_id: str, dialog_id: str) -> bool:
    """Deduplicate retransmitted inbound INVITEs/dialog callbacks."""
    now = time.time()
    key = (pbx_id, dialog_id)
    with _inbound_dialogs_lock:
        stale = [
            existing
            for existing, ts in _inbound_dialogs.items()
            if now - ts > _INBOUND_DIALOG_TTL
        ]
        for existing in stale:
            _inbound_dialogs.pop(existing, None)
        if key in _inbound_dialogs:
            return False
        _inbound_dialogs[key] = now
        return True


def _extract_request_call_id(voip_call) -> str:
    req = getattr(voip_call, "request", None)
    headers = getattr(req, "headers", {}) if req is not None else {}
    if not isinstance(headers, dict):
        return ""
    for name, value in headers.items():
        if str(name).lower() == "call-id":
            if isinstance(value, dict):
                return str(value.get("raw", "") or value.get("value", "")).strip()
            return str(value).strip()
    return ""


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

    # Determine local IP by connecting to the SIP server
    import socket as _sock
    _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
    try:
        _s.connect((sip_proxy, sip_port))
        local_ip = _s.getsockname()[0]
    finally:
        _s.close()

    phone = VoIPPhone(
        server=sip_proxy,
        port=sip_port,
        username=sip_user,
        password=sip_pass,
        callCallback=_on_incoming,
        sipPort=local_port,
        myIP=local_ip,
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


def get_phone(pbx_id: str):
    """Get the VoIPPhone instance for a PBX (for outbound calls)."""
    return _phones.get(pbx_id)


def stop_all() -> None:
    for pbx_id in list(_phones):
        stop_listener(pbx_id)


def _handle_incoming(pbx_id: str, voip_call) -> None:
    """Handle one incoming SIP call."""
    from pyVoIP.VoIP.VoIP import CallState

    sip_call_id = _extract_request_call_id(voip_call)
    if sip_call_id and not _claim_inbound_dialog(pbx_id, sip_call_id):
        _LOGGER.info("Ignoring duplicate inbound SIP callback on %s (call-id=%s)",
                     pbx_id, sip_call_id)
        return

    # Extract caller/callee
    try:
        req = voip_call.request
        caller = req.headers.get("From", {}).get("number", "unknown")
        callee = req.headers.get("To", {}).get("number", "unknown")
    except Exception:
        caller = "unknown"
        callee = "unknown"

    _LOGGER.info("Incoming SIP on %s: %s → %s", pbx_id, caller, callee)

    # Find subscriber by DID, fallback to any subscriber on this PBX
    sub = sub_mod.find_by_did(callee)
    if not sub:
        sub = sub_mod.find_by_pbx(pbx_id)
    if not sub:
        _LOGGER.warning("No subscriber for DID %s on PBX %s — rejecting",
                        callee, pbx_id)
        try:
            voip_call.deny()
        except Exception:
            pass
        return

    # Verify subscriber's account is allowed on this PBX
    from . import auth as auth_mod
    if not auth_mod.check_pbx_access(sub["account_id"], pbx_id):
        _LOGGER.warning("Account %s not allowed on PBX %s — rejecting",
                        sub["account_id"], pbx_id)
        try:
            voip_call.deny()
        except Exception:
            pass
        return

    # Do NOT answer yet — CRM controls when to answer via POST /api/legs/{id}/answer.
    # For pyVoIP: we must answer to get RTP flowing (pyVoIP limitation).
    # For sip_stack trunk: 183 Session Progress already sent, RTP flows as Early Media.
    try:
        voip_call.answer()  # TODO: replace with 183 when pyVoIP supports it
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

    # Fire incoming event asynchronously — CRM handles everything via REST API.
    # Must be async because the CRM makes API calls back to us during handling.
    threading.Thread(
        target=dispatcher.fire_event,
        args=(_EventContext(sub, leg), "incoming",
              {"caller": caller, "callee": callee, "leg_id": leg.leg_id}),
        daemon=True,
    ).start()

    # Wait for CRM to bridge this leg via the API, or timeout.
    # After bridging, bridge_to_call's monitor thread handles
    # cleanup + completed callback. No _monitor_hangup needed.
    _wait_for_bridge(leg, voip_call)


def _wait_for_bridge(leg, voip_call) -> None:
    """Wait for subscriber to bridge the leg via API, or timeout."""
    deadline = time.time() + RING_TIMEOUT
    while time.time() < deadline:
        if leg.call_id:
            return  # bridged via API
        if voip_call is not None:
            try:
                from pyVoIP.VoIP.VoIP import CallState
                if voip_call.state == CallState.ENDED:
                    leg.status = "completed"
                    leg_mod.delete_leg(leg.leg_id)
                    return
            except Exception:
                pass
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


def _monitor_hangup(leg, voip_call) -> None:
    """Block until the SIP call ends, then fire the completed callback."""
    from pyVoIP.VoIP.VoIP import CallState

    while True:
        try:
            if voip_call.state == CallState.ENDED:
                break
        except Exception:
            break
        time.sleep(0.5)

    _LOGGER.info("Inbound leg %s: SIP call ended", leg.leg_id)
    leg.status = "completed"

    # Fire completed callback
    cb_path = leg.callbacks.get("completed")
    if cb_path:
        from . import leg as leg_mod
        leg_mod._fire_callback(leg, "completed")

    # Fire call_ended event to subscriber
    if leg.call_id:
        from . import call_state, subscriber as sub_mod
        call = call_state.get_call(leg.call_id)
        if call:
            dispatcher.fire_event(call, "call_ended",
                                  {"callId": call.call_id})
            call.status = "completed"
            call_state.delete_call(call.call_id)


def _handle_trunk_call(pbx_id: str, caller: str, callee: str,
                       sip_msg: dict, addr, rtp_port: int) -> None:
    """Handle incoming call from trunk via built-in SIP stack (no pyVoIP)."""
    from . import sip_stack

    sip_call_id = sip_stack._get_header(sip_msg, "call-id")
    if sip_call_id and not _claim_inbound_dialog(pbx_id, sip_call_id):
        _LOGGER.info("Ignoring duplicate inbound trunk callback on %s (call-id=%s)",
                     pbx_id, sip_call_id)
        return

    _LOGGER.info("Trunk call on %s: %s → %s", pbx_id, caller, callee)

    # Find subscriber
    sub = sub_mod.find_by_did(callee)
    if not sub:
        sub = sub_mod.find_by_pbx(pbx_id)
    if not sub:
        _LOGGER.warning("No subscriber for DID %s on PBX %s", callee, pbx_id)
        return

    from . import auth as auth_mod
    if not auth_mod.check_pbx_access(sub["account_id"], pbx_id):
        _LOGGER.warning("Account %s not allowed on PBX %s", sub["account_id"], pbx_id)
        return

    # Create RTPSession for inbound audio (Early Media capable)
    from speech_pipeline.RTPSession import RTPSession, RTPCallSession
    from speech_pipeline.rtp_codec import codec_for_pt

    remote_host, remote_port, remote_pt = sip_stack._parse_sdp(sip_msg.get("body", ""))
    codec = codec_for_pt(remote_pt)
    rtp = RTPSession(rtp_port, remote_host, remote_port, codec=codec)
    rtp.start()
    session = RTPCallSession(rtp)

    # Create inbound leg with RTPSession as voip_call
    leg = leg_mod.create_leg(
        direction="inbound",
        number=caller,
        pbx_id=pbx_id,
        subscriber_id=sub["id"],
        voip_call=rtp,  # RTPSession mimics pyVoIP call interface
    )
    leg.status = "ringing"
    leg._sip_msg = sip_msg
    leg._sip_addr = addr
    leg._rtp_port = rtp_port
    leg._rtp_session = rtp
    leg._sip_session = session  # so pipe_executor uses RTPCallSession directly
    # Store SIP Call-ID for deferred answer (Early Media → 200 OK)
    leg._sip_call_id = sip_stack._get_header(sip_msg, "call-id")
    if leg._sip_call_id in sip_stack._trunk_dialogs:
        sip_stack._trunk_dialogs[leg._sip_call_id]["session"] = session
        sip_stack._trunk_dialogs[leg._sip_call_id]["rtp_session"] = rtp
        sip_stack._trunk_dialogs[leg._sip_call_id]["leg_id"] = leg.leg_id

    # Fire incoming event asynchronously
    threading.Thread(
        target=dispatcher.fire_event,
        args=(_EventContext(sub, leg), "incoming",
              {"caller": caller, "callee": callee, "leg_id": leg.leg_id}),
        daemon=True,
    ).start()

    # Wait for CRM to bridge this leg via the API
    _wait_for_bridge(leg, None)


class _EventContext:
    """Minimal object that looks like a Call for fire_event()."""
    def __init__(self, sub: dict, leg):
        self.call_id = leg.leg_id  # use leg_id as event context
        self.subscriber_id = sub["id"]
        self.events = sub.get("events", {})
