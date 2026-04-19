"""SIP leg management using ConferenceMixer.

A leg is a SIP channel. When bridged to a conference:
- SIPSource → pipe() → ConferenceLeg → pipe() → SIPSink
- All format conversion via pipe(). No manual AudioTee/MixMinus.

WEBHOOK CONTRACT
================
The "completed" callback MUST fire exactly once for every leg that
reaches "answered" or "in-progress" status, regardless of how the
leg ends (remote BYE, API delete, call teardown).

Rules:
1. Every monitor thread fires the "completed" callback unconditionally
   when the leg ends — even if ``delete_call`` already set status.
2. The ``_callback_fired`` flag prevents duplicate delivery.
3. ``hangup()`` NEVER fires callbacks — only monitors do.
4. ``delete_leg()`` NEVER fires callbacks — only monitors do.
5. If a leg is deleted before a monitor is started (e.g. ringing leg
   that gets cancelled), the caller (API/originate) is responsible
   for firing the appropriate callback (e.g. "failed").
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import requests as http_requests
from .id_scope import localize_fields, scoped_id

_LOGGER = logging.getLogger("telephony.leg")

_legs: Dict[str, "Leg"] = {}


class Leg:
    """One SIP channel."""

    def __init__(self, leg_id: str, direction: str, number: str,
                 pbx_id: str, subscriber_id: str,
                 voip_call=None):
        self.leg_id = leg_id
        self.direction = direction
        self.number = number
        self.pbx_id = pbx_id
        self.subscriber_id = subscriber_id
        self.status = "ringing"
        self.voip_call = voip_call
        self.call_id: Optional[str] = None
        self.caller_id: Optional[str] = None  # display name for remote party
        self.callbacks: Dict[str, str] = {}
        self.created_at = time.time()
        self.answered_at: Optional[float] = None

        self._src_id: Optional[str] = None
        self._sink_id: Optional[str] = None
        self.sip_session = None
        self.completion_monitor_started = False
        self._callback_fired: Dict[str, bool] = {}  # event -> fired

    def to_dict(self) -> dict:
        return {
            "leg_id": self.leg_id,
            "direction": self.direction,
            "number": self.number,
            "pbx_id": self.pbx_id,
            "subscriber_id": self.subscriber_id,
            "status": self.status,
            "call_id": self.call_id,
            "caller_id": self.caller_id,
            "created_at": self.created_at,
        }

    def hangup(self) -> None:
        """Hang up the SIP leg. Does NOT fire any callbacks (see contract)."""
        # 1. Send SIP BYE FIRST (before closing sockets)
        sent_signaling = False
        if hasattr(self, 'sip_call') and self.sip_call:
            try:
                from . import sip_stack
                sip_stack.hangup(self.sip_call)
                sent_signaling = True
            except Exception:
                pass
        elif hasattr(self, 'sip_call_id') and self.sip_call_id:
            try:
                from . import sip_stack
                sent_signaling = sip_stack.hangup_trunk_leg(self.sip_call_id)
            except Exception:
                pass
        if self.voip_call:
            try:
                if not sent_signaling:
                    self.voip_call.hangup()
            except Exception:
                pass
        if self.sip_session:
            try:
                self.sip_session.hangup()
            except Exception:
                pass
        # 2. Stop RTP
        if hasattr(self, 'rtp_session') and self.rtp_session:
            try:
                self.rtp_session.stop()
            except Exception:
                pass
        # 3. Remove from mixer
        if self._src_id and self.call_id:
            from . import call_state
            call = call_state.get_call(self.call_id)
            if call:
                call.mixer.kill_source(self._src_id)
                if self._sink_id:
                    call.mixer.remove_sink(self._sink_id)
        self.status = "completed"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def create_leg(direction: str, number: str, pbx_id: str,
               subscriber_id: str, voip_call=None) -> Leg:
    owner = subscriber_id
    try:
        from . import subscriber as sub_mod
        sub = sub_mod.get(subscriber_id)
        if sub and sub.get("account_id"):
            owner = sub["account_id"]
    except Exception:
        pass
    leg_id = scoped_id(owner, "leg", entropy_bytes=12)
    leg = Leg(leg_id, direction, number, pbx_id, subscriber_id,
              voip_call=voip_call)
    _legs[leg_id] = leg
    _LOGGER.info("Leg created: %s (%s %s pbx=%s)",
                 leg_id, direction, number, pbx_id)
    return leg


def get_leg(leg_id: str) -> Optional[Leg]:
    return _legs.get(leg_id)


def delete_leg(leg_id: str) -> bool:
    """Remove leg from registry and hang up. Does NOT fire callbacks."""
    leg = _legs.pop(leg_id, None)
    if leg:
        leg.hangup()
        return True
    return False


def remote_end_detected(leg: Leg, session) -> bool:
    """Return True only when the remote SIP side actually ended the leg.

    Local teardown paths such as ``DELETE /api/calls/<id>`` or
    ``DELETE bridge:LEG`` also transition the leg to ``completed``, but
    must not synthesize a second ``completed`` webhook back to the CRM.
    """
    if leg.voip_call and hasattr(leg.voip_call, "state"):
        try:
            from pyVoIP.VoIP.VoIP import CallState
            if leg.voip_call.state == CallState.ENDED:
                return True
        except Exception:
            pass
    if hasattr(leg, "sip_call") and leg.sip_call:
        if leg.sip_call.state == "ended":
            return True
    if session.hungup.is_set():
        return True
    return False


def list_legs(subscriber_id: Optional[str] = None) -> List[Leg]:
    if subscriber_id:
        return [l for l in _legs.values() if l.subscriber_id == subscriber_id]
    return list(_legs.values())


def originate_only(leg: Leg, pbx_entry: dict) -> None:
    """Originate an outbound SIP call — fires callbacks, does NOT bridge.

    The CRM bridges the leg into the conference via POST /pipes
    after receiving the 'answered' callback. This keeps originate
    non-blocking and DSL-compatible.
    """
    fire_callback(leg, "ringing")

    try:
        session = _create_sip_session(leg, pbx_entry)
    except Exception as e:
        _LOGGER.warning("Originate %s failed: %s", leg.number, e)
        leg.status = "failed"
        fire_callback(leg, "failed", error=str(e))
        delete_leg(leg.leg_id)
        return

    # Store session on leg so pipe_executor can use it later
    leg.voip_call = session.call
    leg.sip_session = session
    leg.status = "answered"
    leg.answered_at = time.time()
    fire_callback(leg, "answered")

    # Monitor for end-of-call — fire "completed" callback when done.
    # This monitor runs for legs that are NOT wired via pipe_executor
    # (pipe_executor has its own monitor in _start_sip_monitors).
    def _monitor():
        ended = False
        while leg.status == "answered" or leg.status == "in-progress":
            ended = remote_end_detected(leg, session)
            if ended:
                break
            time.sleep(0.5)

        if not ended:
            _LOGGER.info("Leg %s monitor stopped without remote hangup", leg.leg_id)
            return

        leg.status = "completed"
        duration = time.time() - leg.answered_at if leg.answered_at else 0
        fire_callback(leg, "completed", duration=duration)
        _LOGGER.info("Leg %s ended (duration=%.1fs)", leg.leg_id, duration)
        # Cleanup mixer sources/sinks
        if leg._src_id and leg.call_id:
            from . import call_state
            call = call_state.get_call(leg.call_id)
            if call:
                call.mixer.kill_source(leg._src_id)
                if leg._sink_id:
                    call.mixer.remove_sink(leg._sink_id)
        delete_leg(leg.leg_id)

    leg.completion_monitor_started = True
    threading.Thread(target=_monitor, daemon=True,
                     name=f"mon-{leg.leg_id}").start()


# ---------------------------------------------------------------------------
# SIP session helpers
# ---------------------------------------------------------------------------

def _create_sip_session(leg: Leg, pbx_entry: dict):
    """Create a SIP session — returns a session object with .call, .connected, .hungup.

    For registered SIP devices: uses sip_stack signaling + RTPSession for media.
    For phone numbers: uses sip_stack trunk (INVITE via PBX to PSTN).
    """
    from . import sip_stack

    if not sip_stack.is_running():
        raise RuntimeError("SIP stack not running")

    # CRM-local SIP identities route to registered devices.
    if "@" in leg.number:
        from urllib.parse import unquote
        sip_user = unquote(leg.number)
        if sip_stack.is_local_sip_user(sip_user):
            return _create_local_sip_session(leg, sip_user)
        return _create_trunk_call_session(leg)

    # Phone number → route via trunk to PSTN
    return _create_trunk_call_session(leg)


def _create_local_sip_session(leg: Leg, sip_user: str):
    """Originate via sip_stack directly to all active contacts of a local SIP identity."""
    from . import sip_stack

    regs = sip_stack.get_registrations(sip_user)
    if not regs:
        raise RuntimeError(f"SIP device {sip_user} not registered")
    _LOGGER.info("Originate %s to %d registered contact(s)", leg.number, len(regs))
    sip_call = sip_stack.call_registered_user(sip_user)
    return _wait_and_setup_rtp(leg, sip_call)


def _create_trunk_call_session(leg: Leg):
    """Originate via sip_stack trunk to PSTN."""
    from . import sip_stack

    caller_id = leg.caller_id or leg.callbacks.get("caller_id", "")
    _LOGGER.info("Originate %s via trunk on PBX %s (caller_id=%s)",
                 leg.number, leg.pbx_id, caller_id or "default")
    sip_call = sip_stack.call(leg.pbx_id, leg.number, caller_id=caller_id)
    return _wait_and_setup_rtp(leg, sip_call)


def _wait_and_setup_rtp(leg: Leg, sip_call):
    """Wait for SIP answer, then create RTPSession + RTPCallSession."""
    from . import sip_stack
    from speech_pipeline.RTPSession import RTPSession, RTPCallSession
    from speech_pipeline.rtp_codec import codec_for_pt, PCMU

    leg.sip_call = sip_call  # set IMMEDIATELY so hangup() can CANCEL during ringing

    # Wait for answer (up to 30s)
    deadline = time.time() + 30
    while sip_call.state not in ("answered", "ended"):
        if time.time() > deadline:
            sip_stack.hangup(sip_call)
            raise RuntimeError(f"SIP call to {leg.number} ring timeout")
        sip_call.state_event.wait(timeout=1.0)
        sip_call.state_event.clear()

    if sip_call.state == "ended":
        raise RuntimeError(f"SIP call to {leg.number} ended/rejected")

    # Use codec negotiated from SDP answer
    codec = codec_for_pt(sip_call.negotiated_pt) or PCMU
    _LOGGER.info("SIP answered: %s → RTP %s:%d (local :%d, codec=%s)",
                 leg.number, sip_call.remote_rtp_host,
                 sip_call.remote_rtp_port, sip_call.local_rtp_port, codec)

    rtp = RTPSession(sip_call.local_rtp_port,
                     sip_call.remote_rtp_host, sip_call.remote_rtp_port,
                     codec=codec)
    rtp.start()
    session = RTPCallSession(rtp)
    leg.sip_call = sip_call
    leg.rtp_session = rtp

    # Monitor sip_stack call state → set hungup when ended
    def _monitor():
        while sip_call.state != "ended":
            sip_call.state_event.wait(timeout=2.0)
            sip_call.state_event.clear()
        rtp.stop()
        session.hungup.set()

    threading.Thread(target=_monitor, daemon=True,
                     name=f"mon-{leg.leg_id}").start()
    return session


# ---------------------------------------------------------------------------
# pyVoIP compat
# ---------------------------------------------------------------------------

class PyVoIPCallSession:
    """Wrap a pyVoIP Call so that SIPSource/SIPSink can treat it
    identically to an RTPCallSession."""

    def __init__(self, call):
        self._call = call
        self.connected = threading.Event()
        self.connected.set()  # pyVoIP calls are already connected
        self.hungup = threading.Event()

        import queue
        self.rx_queue: queue.Queue = queue.Queue(maxsize=10)

        self._rx_pump_running = True
        import audioop

        # Detect codec from pyVoIP call — A-law or µ-law
        decode = audioop.ulaw2lin  # default: µ-law
        try:
            from pyVoIP.RTP import PayloadType
            if call.RTPClients and call.RTPClients[0].preference == PayloadType.PCMA:
                decode = audioop.alaw2lin
                _LOGGER.info("PyVoIPCallSession: using A-law decoder")
            else:
                _LOGGER.info("PyVoIPCallSession: using µ-law decoder")
        except Exception:
            pass

        def _rx_pump():
            while self._rx_pump_running:
                try:
                    data = self._call.read_audio(160, blocking=True)
                    if data and len(data) == 160:
                        pcm = decode(data, 2)
                        try:
                            self.rx_queue.put_nowait(pcm)
                        except queue.Full:
                            pass
                except Exception as e:
                    if not self.hungup.is_set():
                        _LOGGER.warning("rx_pump error: %s", e)
                        break

        threading.Thread(target=_rx_pump, daemon=True,
                         name="pyvoip-rx").start()

    def hangup(self):
        """Stop the rx pump and signal hangup."""
        self._rx_pump_running = False
        self.hungup.set()
        try:
            self._call.hangup()
        except Exception:
            pass

    @property
    def call(self):
        return self._call


def fire_callback(leg: Leg, event: str, **extra) -> list:
    """Fire a webhook callback for a leg event. Deduplicates per event."""
    # Exactly-once: skip if this event was already fired for this leg
    if leg._callback_fired.get(event):
        _LOGGER.debug("_fire_callback: %s already fired for leg %s, skipping",
                      event, leg.leg_id)
        return []
    leg._callback_fired[event] = True

    from . import subscriber as sub_mod

    cb_path = leg.callbacks.get(event)
    if not cb_path:
        _LOGGER.warning("_fire_callback: no callback for event=%s (leg=%s, callbacks=%s)",
                        event, leg.leg_id, list(leg.callbacks.keys()))
        return []

    sub = sub_mod.get(leg.subscriber_id)
    if not sub:
        _LOGGER.warning("_fire_callback: no subscriber %s for event=%s",
                        leg.subscriber_id, event)
        return []

    from . import _shared
    url = _shared.subscriber_url(sub, cb_path)
    payload = {
        "leg_id": leg.leg_id,
        "event": event,
        "number": leg.number,
        "direction": leg.direction,
        "call_id": leg.call_id,
        "caller_id": leg.caller_id,
        **extra,
    }
    payload = localize_fields(payload, sub.get("account_id"), "leg_id", "call_id")

    _LOGGER.info("Leg callback %s → %s", event, url)
    _shared.post_webhook(url, payload, sub["bearer_token"])
    return []
