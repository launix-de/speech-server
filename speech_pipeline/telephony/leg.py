"""SIP leg management using tts-piper Stage pipelines.

A leg is a SIP channel.  When bridged to a conference, two Stage
pipelines are built with mix-minus (no self-echo):

    RX: SIPSource → u8→s16le → AudioTee ──→ QueueSink(mixer)
                                    └──→ own_queue

    TX: QueueSource(mixer_output) → MixMinus(- own) → s16le→u8 → SIPSink
"""
from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from typing import Dict, List, Optional

import requests as http_requests

from speech_pipeline.QueueSink import QueueSink
from speech_pipeline.QueueSource import QueueSource
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

        # Stage pipelines (set when bridged)
        self._rx_pipeline: Optional[QueueSink] = None
        self._tx_pipeline: Optional[SIPSink] = None
        self._threads: List[threading.Thread] = []
        self._sip_session = None  # SIPSession for outbound legs

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
        if self._rx_pipeline:
            self._rx_pipeline.cancel()
        if self._tx_pipeline:
            self._tx_pipeline.cancel()
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
# Bridge: connect a SIP leg to a conference mixer via Stage pipelines
# ---------------------------------------------------------------------------

def bridge_to_call(leg: Leg, call) -> None:
    """Bridge leg audio into call.mixer → call.tee with mix-minus.

    Pipelines (all Stage-based)::

        RX: SIPSource → u8→s16le → AudioTee ──→ QueueSink(call.mixer input)
                                        └──→ own_queue (for MixMinus)

        TX: call.tee sidechain → MixMinus(- own_queue) → s16le→u8 → SIPSink
    """
    from speech_pipeline.AudioTee import AudioTee
    from speech_pipeline.MixMinus import MixMinus

    leg.call_id = call.call_id
    leg.status = "in-progress"
    leg.answered_at = time.time()
    call.register_participant(leg.leg_id, type="sip",
                              direction=leg.direction, number=leg.number)

    session = _CallSession(leg.voip_call)

    # RX: SIP → tee → (1) mixer input + (2) own_queue for MixMinus
    mixer_input_q = call.mixer.add_input()
    own_queue = queue.Queue(maxsize=200)

    rx_source = SIPSource(session)
    rx_tee = AudioTee(8000, "s16le")
    rx_tee.add_mixer_feed(own_queue)                          # copy for MixMinus
    rx_sink = QueueSink(mixer_input_q, sample_rate=8000, encoding="s16le")
    rx_source.pipe(rx_tee).pipe(rx_sink)                      # u8→s16le auto-inserted
    leg._rx_pipeline = rx_sink

    # TX: call.tee → MixMinus(- own) → SIP
    mix_minus = MixMinus(sample_rate=8000)
    mix_minus.set_own(own_queue)
    tx_sink = SIPSink(session)
    mix_minus.pipe(tx_sink)                                   # s16le→u8 auto-inserted
    call.tee.add_sidechain(mix_minus)                         # attaches to conference tee
    leg._tx_pipeline = tx_sink

    # Run both pipelines in background threads
    def _run_rx():
        try:
            rx_sink.run()
        except Exception as e:
            _LOGGER.warning("Leg %s RX error: %s", leg.leg_id, e)

    def _run_tx():
        try:
            tx_sink.run()
        except Exception as e:
            _LOGGER.warning("Leg %s TX error: %s", leg.leg_id, e)

    t_rx = threading.Thread(target=_run_rx, daemon=True,
                            name=f"rx-{leg.leg_id}")
    t_tx = threading.Thread(target=_run_tx, daemon=True,
                            name=f"tx-{leg.leg_id}")
    t_rx.start()
    t_tx.start()
    leg._threads = [t_rx, t_tx]

    # Monitor for SIP hangup
    def _monitor():
        from pyVoIP.VoIP.VoIP import CallState
        while leg.status == "in-progress":
            if leg.voip_call.state == CallState.ENDED:
                break
            time.sleep(0.5)

        leg.status = "completed"
        duration = time.time() - leg.answered_at if leg.answered_at else 0

        # Cancel pipelines
        rx_source.cancel()
        tx_source.cancel()
        call.remove_input(leg.leg_id)

        t_rx.join(timeout=3)
        t_tx.join(timeout=3)

        _fire_callback(leg, "completed", duration=duration)
        _LOGGER.info("Leg %s ended (duration=%.1fs)", leg.leg_id, duration)

    threading.Thread(target=_monitor, daemon=True,
                     name=f"mon-{leg.leg_id}").start()


def originate_and_bridge(leg: Leg, call, pbx_entry: dict) -> None:
    """Originate an outbound SIP call and bridge into conference.

    Uses SIPSession to dial out, then bridges via ``bridge_to_call``.
    """
    from speech_pipeline.SIPSession import SIPSession

    _fire_callback(leg, "ringing")

    try:
        session = SIPSession(
            target=leg.number,
            server=pbx_entry.get("sip_proxy", "127.0.0.1"),
            port=int(pbx_entry.get("sip_port", 5060)),
            username=pbx_entry.get("sip_user", "piper"),
            password=pbx_entry.get("sip_password", ""),
        )
        session.start()  # blocks until answered or timeout
    except Exception as e:
        _LOGGER.warning("Originate %s failed: %s", leg.number, e)
        leg.status = "failed"
        _fire_callback(leg, "failed", error=str(e))
        delete_leg(leg.leg_id)
        return

    leg.voip_call = session.call
    leg._sip_session = session
    bridge_to_call(leg, call)
    _fire_callback(leg, "answered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CallSession:
    """Minimal adapter so SIPSource/SIPSink can use a raw pyVoIP call."""

    def __init__(self, voip_call):
        self._call = voip_call
        self.connected = threading.Event()
        self.hungup = threading.Event()
        self.connected.set()  # already answered

    @property
    def call(self):
        return self._call


def _fire_callback(leg: Leg, event: str, **extra) -> list:
    from . import subscriber as sub_mod

    cb_path = leg.callbacks.get(event)
    if not cb_path:
        return []

    sub = sub_mod.get(leg.subscriber_id)
    if not sub:
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
        resp = http_requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {sub['bearer_token']}",
        }, timeout=10)
        if resp.status_code == 200 and resp.content:
            return resp.json().get("commands", [])
    except Exception as e:
        _LOGGER.warning("Leg callback %s → %s failed: %s", event, url, e)
    return []
