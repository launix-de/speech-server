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
    """Bridge SIP leg into conference via raw mixer queues.

    Same pattern as webclient: add_participant() for raw queue I/O.
    RX/TX pump threads do inline format conversion (8k↔48k, codec).
    No SIPSource/SIPSink/pipe() — no extra Stage threads.
    """
    import audioop
    import queue as _queue
    from speech_pipeline.RTPSession import RTPSession

    leg.call_id = call.call_id
    leg.status = "in-progress"
    leg.answered_at = time.time()
    call.register_participant(leg.leg_id, type="sip",
                              direction=leg.direction, number=leg.number)

    session = _CallSession(leg.voip_call)
    voip_call = leg.voip_call
    is_rtp = isinstance(voip_call, RTPSession)

    # Raw mixer queues — same as webclient, no Stage overhead
    in_q: _queue.Queue = _queue.Queue(maxsize=50)
    leg._src_id, out_q = call.mixer.add_participant(in_q)

    # RX pump: session.rx_queue → resample 8k→48k → mixer input queue
    def _rx_pump():
        state = None
        while leg.status == "in-progress" and not session.hungup.is_set():
            try:
                frame = session.rx_queue.get(timeout=0.5)
            except _queue.Empty:
                continue
            if not frame:
                continue
            # Decode to s16le if pyVoIP (u8)
            if not is_rtp:
                signed = audioop.bias(frame, 1, -128)
                frame = audioop.lin2lin(signed, 1, 2)
            # Resample 8kHz → 48kHz
            resampled, state = audioop.ratecv(
                frame, 2, 1, 8000, 48000, state)
            try:
                in_q.put_nowait(resampled)
            except _queue.Full:
                pass  # drop — bounded delay

    # TX pump: mixer output queue → resample 48k→8k → SIP send
    def _tx_pump():
        state = None
        # pyVoIP codec setup
        encode = None
        if not is_rtp:
            try:
                from pyVoIP.RTP import PayloadType
                rtp = voip_call.RTPClients[0]
                if rtp.preference == PayloadType.PCMA:
                    encode = lambda pcm: audioop.lin2alaw(pcm, 2)
                else:
                    encode = lambda pcm: audioop.lin2ulaw(pcm, 2)
            except Exception:
                encode = lambda pcm: audioop.lin2ulaw(pcm, 2)

        while leg.status == "in-progress" and not session.hungup.is_set():
            try:
                frame = out_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            if frame is None:
                break
            # Resample 48kHz → 8kHz
            resampled, state = audioop.ratecv(
                frame, 2, 1, 48000, 8000, state)
            # Send to SIP
            if is_rtp:
                voip_call.write_s16le(resampled)
            else:
                voip_call.write_audio(encode(resampled))

    threading.Thread(target=_rx_pump, daemon=True,
                     name=f"rx-{leg.leg_id}").start()
    threading.Thread(target=_tx_pump, daemon=True,
                     name=f"tx-{leg.leg_id}").start()

    _LOGGER.info("Leg %s bridged to call %s (src=%s, raw queues)",
                 leg.leg_id, call.call_id, leg._src_id)

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

    # Monitor for SIP hangup — single cleanup point for all leg types
    def _monitor():
        while leg.status == "in-progress":
            ended = False
            # pyVoIP leg (has .state attribute with CallState enum)
            if leg.voip_call and hasattr(leg.voip_call, 'state'):
                try:
                    from pyVoIP.VoIP.VoIP import CallState
                    if leg.voip_call.state == CallState.ENDED:
                        ended = True
                except Exception:
                    pass  # don't assume ended on error
            # sip_stack device leg
            if hasattr(leg, '_sip_call') and leg._sip_call:
                if leg._sip_call.state == "ended":
                    ended = True
            # session hungup (RTPSession or SIPSession)
            if session.hungup.is_set():
                ended = True
            if ended:
                break
            time.sleep(0.5)

        leg.status = "completed"
        session.hungup.set()  # stop RX/TX pumps
        duration = time.time() - leg.answered_at if leg.answered_at else 0
        try:
            call.mixer.kill_source(leg._src_id)
        except Exception:
            pass
        call.unregister_participant(leg.leg_id)

        _fire_callback(leg, "completed", duration=duration)
        _LOGGER.info("Leg %s ended (duration=%.1fs)", leg.leg_id, duration)
        delete_leg(leg.leg_id)

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

    # Bridge into conference — bridge_to_call spawns a monitor thread
    # that handles cleanup + completed callback when the SIP call ends.
    # We do NOT do our own cleanup here to avoid double-fire.
    leg.voip_call = session.call
    bridge_to_call(leg, call)
    _fire_callback(leg, "answered")


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
    from speech_pipeline.rtp_codec import codec_for_pt, PCMU

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

    # Use codec negotiated from SDP answer
    codec = codec_for_pt(sip_call.negotiated_pt) or PCMU
    _LOGGER.info("SIP answered: %s → RTP %s:%d (local :%d, codec=%s)",
                 leg.number, sip_call.remote_rtp_host,
                 sip_call.remote_rtp_port, sip_call.local_rtp_port, codec)

    # Create RTP media session + wrap as SIPSource/SIPSink-compatible session
    rtp = RTPSession(sip_call.local_rtp_port,
                     sip_call.remote_rtp_host, sip_call.remote_rtp_port,
                     codec=codec)
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
    """Adapter so SIPSource/SIPSink can use a raw pyVoIP call.

    Provides rx_queue (same pattern as AudioSocketSession) —
    a pump thread reads from pyVoIP and fills the queue.
    """

    def __init__(self, voip_call):
        import audioop
        import queue as _queue
        self._call = voip_call
        self.connected = threading.Event()
        self.hungup = threading.Event()
        self.connected.set()
        self.rx_queue: _queue.Queue = _queue.Queue(maxsize=10)

        # Setup pyVoIP RTP: ring buffer + encode bypass + timing fix
        try:
            import pyVoIP
            pyVoIP.TRANSMIT_DELAY_REDUCTION = 1.0
        except ImportError:
            pass

        if hasattr(voip_call, 'RTPClients') and voip_call.RTPClients:
            from speech_pipeline.SIPSink import _RingBuffer
            rtp = voip_call.RTPClients[0]
            try:
                from pyVoIP.RTP import PayloadType
                silence = 0xd5 if rtp.preference == PayloadType.PCMA else 0xff
            except Exception:
                silence = 0xff
            rb = _RingBuffer(silence_byte=silence)
            rtp.pmout = rb
            rtp.encode_packet = lambda data: data

        # Pump: pyVoIP read_audio → rx_queue (bounded, drop on full)
        def _rx_pump():
            while not self.hungup.is_set():
                try:
                    frame = voip_call.read_audio(length=160, blocking=True)
                    if frame:
                        try:
                            self.rx_queue.put_nowait(frame)
                        except _queue.Full:
                            pass  # drop — bounded delay
                except Exception:
                    if not self.hungup.is_set():
                        break

        threading.Thread(target=_rx_pump, daemon=True,
                         name="pyvoip-rx").start()

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

    def _send():
        try:
            resp = http_requests.post(url, json=payload, headers={
                "Authorization": f"Bearer {sub['bearer_token']}",
            }, timeout=10)
            _LOGGER.info("Leg callback %s → %d", event, resp.status_code)
        except Exception as e:
            _LOGGER.warning("Leg callback %s → %s failed: %s", event, url, e)

    # Fire async — don't block the caller
    _LOGGER.info("Leg callback %s → %s", event, url)
    threading.Thread(target=_send, daemon=True).start()
    return []
