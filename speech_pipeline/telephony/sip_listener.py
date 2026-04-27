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
# Some CRM deployments answer the webhook quickly but only create/bridge the
# call graph a bit later. Keep SIP legs alive for a bounded grace window so a
# slightly late /api/calls + /api/pipelines sequence does not lose the leg.
LATE_BRIDGE_GRACE_TIMEOUT = 45

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

    # Skip listeners for PBX entries without SIP credentials — otherwise
    # pyVoIP's ``VoIPPhone.start`` blocks on REGISTER against ``""`` /
    # a nonexistent proxy and deadlocks the caller.
    sip_proxy = pbx_entry.get("sip_proxy") or ""
    sip_user = pbx_entry.get("sip_user") or ""
    if not sip_proxy or not sip_user:
        _LOGGER.info("PBX %s: no sip_proxy/sip_user set — skipping "
                     "in-process SIP listener (built-in sip_stack handles trunking)",
                     pbx_id)
        return

    # If the built-in sip_stack is running it already registers this PBX
    # as a trunk.  Running a second pyVoIP registration in parallel
    # causes both to claim the same inbound INVITE; pyVoIP then decodes
    # the trunk's G.722 payload as A-law (its default) → audio garbage.
    try:
        from . import sip_stack
        if sip_stack.is_running():
            _LOGGER.info(
                "PBX %s: sip_stack is running and handles trunking — "
                "skipping pyVoIP in-process listener to avoid "
                "duplicate registration + codec mismatch",
                pbx_id,
            )
            return
    except Exception:
        pass

    try:
        from pyVoIP.VoIP.VoIP import VoIPPhone
    except ImportError:
        _LOGGER.warning("pyVoIP not installed — SIP listener disabled")
        return

    from speech_pipeline.SIPSession import _find_free_udp_port, _patch_voip_phone

    sip_port = int(pbx_entry.get("sip_port", 5060))
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

    # Find subscriber by DID, fallback to the single wildcard subscriber for
    # this PBX.  This is intentional multi-CRM routing: exactly one CRM owns
    # an inbound dialog, and the speech server must never fan out the same
    # incoming call to multiple CRMs.
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

    # Design intent: the CRM controls when a leg is answered by issuing the
    # explicit ``answer:LEG`` action over the pipeline API.
    #
    # Legacy pyVoIP limitation: this fallback path cannot provide Early Media,
    # so it answers immediately to get RTP flowing at all.  The production
    # built-in SIP stack avoids that compromise by using 183 + deferred 200 OK.
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

    # Fire incoming event asynchronously.  The CRM owns the actual call graph
    # and will assemble it via REST/DSL callbacks back into this server.
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


def _wait_for_bridge(leg, voip_call, *,
                     ring_timeout: float | None = None,
                     grace_timeout: float | None = None) -> None:
    """Wait for subscriber to bridge the leg via API, or timeout.

    ``ring_timeout < 0`` disables the server-side timeout entirely and leaves
    teardown to the remote SIP endpoint / explicit API cleanup. That is useful
    for user-initiated registered-client outbound calls where the CRM may take
    arbitrarily long before it posts the call graph back.
    """
    if ring_timeout is None:
        ring_timeout = RING_TIMEOUT
    if grace_timeout is None:
        grace_timeout = LATE_BRIDGE_GRACE_TIMEOUT

    deadline = time.time() + ring_timeout if ring_timeout >= 0 else None
    hard_deadline = deadline
    grace_logged = False
    session = getattr(leg, "sip_session", None)
    if session is not None and deadline is not None:
        hard_deadline += grace_timeout

    while hard_deadline is None or time.time() < hard_deadline:
        if leg.call_id:
            return  # bridged via API
        if session is not None and hasattr(session, "hungup") and session.hungup.is_set():
            leg.status = "completed"
            leg_mod.delete_leg(leg.leg_id)
            return
        if voip_call is not None:
            try:
                from pyVoIP.VoIP.VoIP import CallState
                if voip_call.state == CallState.ENDED:
                    leg.status = "completed"
                    leg_mod.delete_leg(leg.leg_id)
                    return
            except Exception:
                pass
        if (
            not grace_logged
            and session is not None
            and deadline is not None
            and time.time() >= deadline
        ):
            grace_logged = True
            _LOGGER.warning(
                "Leg %s: no bridge within %ds — keeping SIP leg alive for %ds grace",
                leg.leg_id,
                ring_timeout,
                grace_timeout,
            )
        time.sleep(0.5)

    # Timeout — reject
    total_timeout = ring_timeout + (grace_timeout if session is not None else 0)
    _LOGGER.warning("Leg %s: no bridge within %ds — hanging up",
                    leg.leg_id, total_timeout)
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
        leg_mod.fire_callback(leg, "completed")

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

    sip_call_id = sip_stack.get_header(sip_msg, "call-id")
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

    remote_host, remote_port, remote_pt = sip_stack.parse_sdp(sip_msg.get("body", ""))
    codec = codec_for_pt(remote_pt)
    # Check if remote offers DTLS-SRTP
    remote_fp, remote_setup = sip_stack.parse_sdp_dtls(sip_msg.get("body", ""))
    dtls_role = "server" if remote_fp else None  # we are answerer → server
    rtp = RTPSession(rtp_port, remote_host, remote_port, codec=codec,
                     dtls_role=dtls_role)
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
    leg.sip_msg = sip_msg
    leg.sip_addr = addr
    leg.rtp_port = rtp_port
    leg.rtp_session = rtp
    leg.sip_session = session  # so pipe_executor uses RTPCallSession directly
    # Store SIP Call-ID for deferred answer (Early Media → 200 OK)
    leg.sip_call_id = sip_stack.get_header(sip_msg, "call-id")
    if leg.sip_call_id in sip_stack.trunk_dialogs:
        sip_stack.trunk_dialogs[leg.sip_call_id]["session"] = session
        sip_stack.trunk_dialogs[leg.sip_call_id]["rtp_session"] = rtp
        sip_stack.trunk_dialogs[leg.sip_call_id]["leg_id"] = leg.leg_id

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
