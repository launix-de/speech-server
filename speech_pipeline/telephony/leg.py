"""SIP leg management using ConferenceMixer.

A leg is a SIP channel. When bridged to a conference:
- SIPSource → conf.add_source() (auto-resamples 8kHz u8 → 48kHz s16le)
- conf.add_sink(SIPSink, mute_source=src_id) (auto-resamples 48kHz → 8kHz)

All format conversion via pipe(). No manual AudioTee/MixMinus.
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Dict, List, Optional

import requests as http_requests

from speech_pipeline.SIPSource import SIPSource
from speech_pipeline.SIPSink import SIPSink

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
        self.callbacks: Dict[str, str] = {}
        self.created_at = time.time()
        self.answered_at: Optional[float] = None

        self._src_id: Optional[str] = None
        self._sink_id: Optional[str] = None
        self._sip_session = None

    def to_dict(self) -> dict:
        return {
            "leg_id": self.leg_id,
            "direction": self.direction,
            "number": self.number,
            "pbx_id": self.pbx_id,
            "subscriber_id": self.subscriber_id,
            "status": self.status,
            "call_id": self.call_id,
            "created_at": self.created_at,
        }

    def hangup(self) -> None:
        # 1. Send SIP BYE FIRST (before closing sockets)
        if hasattr(self, '_sip_call') and self._sip_call:
            try:
                from . import sip_stack
                sip_stack.hangup(self._sip_call)
            except Exception:
                pass
        if self.voip_call:
            try:
                self.voip_call.hangup()
            except Exception:
                pass
        if self._sip_session:
            try:
                self._sip_session.hangup()
            except Exception:
                pass
        # 2. Stop RTP
        if hasattr(self, '_rtp_session') and self._rtp_session:
            try:
                self._rtp_session.stop()
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
    leg_id = "leg-" + secrets.token_urlsafe(12)
    leg = Leg(leg_id, direction, number, pbx_id, subscriber_id,
              voip_call=voip_call)
    _legs[leg_id] = leg
    _LOGGER.info("Leg created: %s (%s %s pbx=%s)",
                 leg_id, direction, number, pbx_id)
    return leg


def get_leg(leg_id: str) -> Optional[Leg]:
    return _legs.get(leg_id)


def delete_leg(leg_id: str) -> bool:
    leg = _legs.pop(leg_id, None)
    if leg:
        leg.hangup()
        return True
    return False


def list_legs(subscriber_id: Optional[str] = None) -> List[Leg]:
    if subscriber_id:
        return [l for l in _legs.values() if l.subscriber_id == subscriber_id]
    return list(_legs.values())


# ---------------------------------------------------------------------------
# Bridge: connect a SIP leg to a conference via ConferenceMixer
# ---------------------------------------------------------------------------

def bridge_to_call(leg: Leg, call) -> None:
    """Bridge SIP leg into conference using ConferenceMixer.

    RX: SIPSource → conf.add_source (auto u8@8k → s16le@48k via pipe)
    TX: conf.add_sink(SIPSink, mute_source=src_id) (auto 48k → 8k u8)
    """
    leg.call_id = call.call_id
    leg.status = "in-progress"
    leg.answered_at = time.time()
    call.register_participant(leg.leg_id, type="sip",
                              direction=leg.direction, number=leg.number)

    session = _CallSession(leg.voip_call)

    # RX: SIPSource → conference mixer (auto-resampled)
    rx_source = SIPSource(session)
    leg._src_id = call.mixer.add_source(rx_source)

    # TX: conference mix (minus own) → SIPSink (auto-resampled)
    tx_sink = SIPSink(session)
    leg._sink_id = call.mixer.add_sink(tx_sink, mute_source=leg._src_id)

    _LOGGER.info("Leg %s bridged to call %s (src=%s sink=%s)",
                 leg.leg_id, call.call_id, leg._src_id, leg._sink_id)

    # Monitor for DTMF digits from caller → fire subscriber callback
    def _dtmf_monitor():
        while leg.status == "in-progress":
            try:
                digit = leg.voip_call.get_dtmf(length=1)
                if digit:
                    _LOGGER.info("Leg %s DTMF received: %s", leg.leg_id, digit)
                    _fire_callback(leg, "dtmf", digit=digit,
                                   call_id=call.call_id)
            except Exception:
                time.sleep(0.1)

    threading.Thread(target=_dtmf_monitor, daemon=True,
                     name=f"dtmf-{leg.leg_id}").start()

    # Monitor for SIP hangup
    def _monitor():
        from pyVoIP.VoIP.VoIP import CallState
        while leg.status == "in-progress":
            if leg.voip_call.state == CallState.ENDED:
                break
            time.sleep(0.5)

        leg.status = "completed"
        duration = time.time() - leg.answered_at if leg.answered_at else 0
        call.mixer.kill_source(leg._src_id)
        if leg._sink_id:
            call.mixer.remove_sink(leg._sink_id)
        call.unregister_participant(leg.leg_id)

        _fire_callback(leg, "completed", duration=duration)
        _LOGGER.info("Leg %s ended (duration=%.1fs)", leg.leg_id, duration)

    threading.Thread(target=_monitor, daemon=True,
                     name=f"mon-{leg.leg_id}").start()


def originate_and_bridge(leg: Leg, call, pbx_entry: dict) -> None:
    """Originate an outbound SIP call and bridge into conference.

    Routing:
    - SIP URI with @ + registered device → sip_stack direct INVITE + RTPSession
    - Phone number or unregistered → pyVoIP trunk (SIPSession)

    Both paths produce a session compatible with SIPSource/SIPSink,
    then feed through the same bridge_to_call → ConferenceMixer pipeline.
    """
    _fire_callback(leg, "ringing")

    try:
        session = _create_sip_session(leg, pbx_entry)
    except Exception as e:
        _LOGGER.warning("Originate %s failed: %s", leg.number, e)
        leg.status = "failed"
        _LOGGER.info("Firing failed callback for leg %s (callbacks=%s, sub=%s)",
                     leg.leg_id, list(leg.callbacks.keys()), leg.subscriber_id)
        _fire_callback(leg, "failed", error=str(e))
        delete_leg(leg.leg_id)
        return

    # Bridge into conference (identical for all session types)
    leg.voip_call = session.call
    bridge_to_call(leg, call)
    _fire_callback(leg, "answered")

    # Monitor for hangup
    session.hungup.wait()
    _LOGGER.info("Originate %s: call ended", leg.number)
    leg.status = "completed"
    _fire_callback(leg, "completed")
    delete_leg(leg.leg_id)


def _create_sip_session(leg: Leg, pbx_entry: dict):
    """Create a SIP session — returns a session object with .call, .connected, .hungup.

    For registered SIP devices: uses sip_stack signaling + RTPSession for media.
    For phone numbers: uses pyVoIP (full SIP+RTP stack).
    """
    from . import sip_stack

    # Check if target is a registered SIP device
    if sip_stack._running and "@" in leg.number:
        from urllib.parse import unquote
        sip_user = unquote(leg.number)
        reg = sip_stack.get_registration(sip_user)
        if reg:
            return _create_sip_stack_session(leg, reg)
        raise RuntimeError(f"SIP device {sip_user} not registered")

    # Phone number → pyVoIP trunk
    return _create_pyvoip_session(leg, pbx_entry)


def _create_sip_stack_session(leg: Leg, reg: dict):
    """Originate via sip_stack + RTPSession for audio."""
    from . import sip_stack
    from speech_pipeline.RTPSession import RTPSession, RTPCallSession

    _LOGGER.info("Originate %s to device %s", leg.number, reg.get("contact_uri", "?"))
    sip_call = sip_stack.call_device(leg.number, reg)

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

    _LOGGER.info("SIP answered: %s → RTP %s:%d (local :%d)",
                 leg.number, sip_call.remote_rtp_host,
                 sip_call.remote_rtp_port, sip_call.local_rtp_port)

    # Create RTP media session + wrap as SIPSource/SIPSink-compatible session
    rtp = RTPSession(sip_call.local_rtp_port,
                     sip_call.remote_rtp_host, sip_call.remote_rtp_port)
    rtp.start()
    session = RTPCallSession(rtp)
    leg._sip_call = sip_call
    leg._rtp_session = rtp

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


def _create_pyvoip_session(leg: Leg, pbx_entry: dict):
    """Originate via pyVoIP (full SIP+RTP stack)."""
    from speech_pipeline.SIPSession import SIPSession

    _LOGGER.info("Originate %s via pyVoIP on PBX %s", leg.number, leg.pbx_id)
    session = SIPSession(
        target=leg.number,
        server=pbx_entry.get("sip_proxy", "127.0.0.1"),
        port=int(pbx_entry.get("sip_port", 5060)),
        username=pbx_entry.get("sip_user", "piper"),
        password=pbx_entry.get("sip_password", ""),
    )
    session.start()  # Blocks until registered + answered
    return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CallSession:
    """Minimal adapter so SIPSource/SIPSink can use a raw pyVoIP call."""

    def __init__(self, voip_call):
        self._call = voip_call
        self.connected = threading.Event()
        self.hungup = threading.Event()
        self.connected.set()

    @property
    def call(self):
        return self._call


def _fire_callback(leg: Leg, event: str, **extra) -> list:
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

    url = sub["base_url"].rstrip("/") + "/" + cb_path.lstrip("/")
    payload = {
        "leg_id": leg.leg_id,
        "event": event,
        "number": leg.number,
        "direction": leg.direction,
        "call_id": leg.call_id,
        **extra,
    }

    try:
        _LOGGER.info("Leg callback %s → %s", event, url)
        resp = http_requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {sub['bearer_token']}",
        }, timeout=10)
        _LOGGER.info("Leg callback %s → %d %s", event, resp.status_code, resp.text[:200] if resp.text else "")
        if resp.status_code == 200 and resp.content:
            return resp.json().get("commands", [])
    except Exception as e:
        _LOGGER.warning("Leg callback %s → %s failed: %s", event, url, e)
    return []
