"""Codec-aware RTP session — provides the same audio interface as pyVoIP calls.

Used by SIPSource/SIPSink to bridge sip_stack-originated calls into
the Stage pipeline.  Supports any codec via the RTPCodec abstraction.

Interface contract:
- ``read_s16le(blocking)`` → s16le PCM bytes (decoded from wire)
- ``write_s16le(data)``    → encode s16le to wire and queue for TX
- ``write_wire(data)``     → queue pre-encoded wire bytes for TX
- ``codec``                → the negotiated RTPCodec instance

Legacy pyVoIP-compat (used by SIPSink pyVoIP path):
- ``RTPClients[0].pmout``  → output buffer (replaced by SIPSink)
"""
from __future__ import annotations

import audioop
import logging
import queue
import socket
import struct
import threading
import time
from typing import Optional

from speech_pipeline.rtp_codec import PCMU, RTPCodec

_LOGGER = logging.getLogger("rtp-session")

RTP_HEADER_SIZE = 12


class RTPSession:
    """Raw UDP RTP session with codec negotiation.

    After construction, wrap in ``RTPCallSession`` and pass to
    ``SIPSource``/``SIPSink`` — they detect the codec automatically.
    """

    def __init__(self, local_port: int, remote_host: str, remote_port: int,
                 codec: Optional[RTPCodec] = None,
                 dtls_role: Optional[str] = None) -> None:
        self.local_port = local_port
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.codec = codec or PCMU
        self._dtls_role = dtls_role  # None=plain RTP, "server"/"client"=DTLS-SRTP
        self._dtls = None  # DtlsSrtpSession, set in start() if dtls_role

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", local_port))
        self._sock.settimeout(0.1)

        self._rx_queue: queue.Queue = queue.Queue(maxsize=10)
        # Decoded + upsampled frames for SIPSource (48kHz s16le, like CodecSocketSession)
        self.rx_queue: queue.Queue = queue.Queue(maxsize=10)
        self._rx_resample_state = None
        self._tx_queue: queue.Queue = queue.Queue(maxsize=10)
        self._running = False
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None

        # TX state
        self._tx_seq = 0
        self._tx_ts = 0
        self._tx_ssrc = struct.unpack("!I", struct.pack("!I", id(self) & 0xFFFFFFFF))[0]

        # Fake RTPClient for SIPSink pyVoIP compatibility
        self._rtp_client = _RTPClient(self)
        self.RTPClients = [self._rtp_client]

        _LOGGER.info("RTPSession %s: %s:%d ↔ %s:%d",
                      self.codec, "0.0.0.0", local_port, remote_host, remote_port)

    def start(self) -> None:
        """Start RTP receive and transmit threads.

        If dtls_role is set, performs DTLS handshake first (blocking).
        Falls back to plain RTP if handshake fails.
        """
        self._running = True

        # Optional DTLS-SRTP handshake
        if self._dtls_role:
            try:
                from .dtls_srtp import DtlsSrtpSession
                self._dtls = DtlsSrtpSession(
                    self._sock,
                    role=self._dtls_role,
                    remote_addr=(self.remote_host, self.remote_port),
                )
                if not self._dtls.do_handshake(timeout=5.0):
                    _LOGGER.warning("DTLS handshake failed, falling back to plain RTP")
                    self._dtls = None
            except Exception as exc:
                _LOGGER.warning("DTLS init failed (%s), falling back to plain RTP", exc)
                self._dtls = None

        self._rx_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="rtp-rx")
        self._rx_thread.start()
        self._tx_thread = threading.Thread(
            target=self._tx_loop, daemon=True, name="rtp-tx")
        self._tx_thread.start()

    def stop(self) -> None:
        """Stop the session and close the socket."""
        self._running = False
        try:
            self.codec.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    # ----- Audio API (s16le) -----

    def read_s16le(self, blocking: bool = False) -> bytes:
        """Read one frame from RTP, decoded to s16le PCM."""
        try:
            timeout = 0.02 if blocking else 0.0
            wire_data = self._rx_queue.get(timeout=timeout)
        except queue.Empty:
            return b""
        if not wire_data:
            return b""
        return self.codec.decode(wire_data)

    def write_s16le(self, data: bytes) -> None:
        """Queue s16le PCM for encoding and TX."""
        try:
            self._tx_queue.put_nowait(data)
        except queue.Full:
            pass  # drop — real-time audio, don't build up delay

    # ----- Legacy pyVoIP-compat API (u8) -----

    def read_audio(self, length: int = 160, blocking: bool = False) -> bytes:
        """Read audio decoded to s16le (kept for SIPSource compat).

        Note: despite the name, this now returns s16le, not u8.
        SIPSource detects RTPSession and sets output_format accordingly.
        """
        return self.read_s16le(blocking=blocking)

    # ----- TX loop -----

    def _tx_loop(self) -> None:
        """Send RTP packets with codec pacing.

        Receives s16le PCM via _tx_queue, segments into codec frames,
        encodes, and sends at real-time cadence.
        """
        pcm_frame_bytes = self.codec.frame_samples * 2  # s16le = 2 bytes/sample
        frame_duration = self.codec.frame_samples / float(self.codec.sample_rate)
        buf = b""
        next_send = time.monotonic()

        while self._running:
            # Block until data arrives
            try:
                chunk = self._tx_queue.get(timeout=0.5)
                buf += chunk
            except queue.Empty:
                continue

            # Drain any additional queued data
            try:
                while True:
                    buf += self._tx_queue.get_nowait()
            except queue.Empty:
                pass

            # Segment PCM into codec frames, encode, and send.
            while len(buf) >= pcm_frame_bytes:
                now = time.monotonic()
                if next_send > now:
                    time.sleep(next_send - now)
                elif now - next_send > frame_duration * 4:
                    next_send = now
                wire = self.codec.encode(buf[:pcm_frame_bytes])
                buf = buf[pcm_frame_bytes:]
                if wire:
                    packet = self._build_rtp_packet(wire)
                    # SRTP encrypt (if DTLS-SRTP is active)
                    if self._dtls and self._dtls.encrypted:
                        packet = self._dtls.protect(packet)
                    try:
                        self._sock.sendto(packet, (self.remote_host, self.remote_port))
                    except Exception:
                        pass
                self._tx_seq = (self._tx_seq + 1) & 0xFFFF
                self._tx_ts += self.codec.timestamp_step
                next_send += frame_duration

    def hangup(self) -> None:
        """Stop the session (called by leg cleanup)."""
        self.stop()

    def get_dtmf(self, _=None) -> str:
        """DTMF stub — not implemented for raw RTP."""
        return ""

    # ----- internal -----

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                continue

            # DTLS packets (content type 20-63) go to the handshake handler
            if self._dtls and len(data) > 0 and 20 <= data[0] <= 63:
                self._dtls.feed_dtls(data)
                continue

            # SRTP decrypt (if DTLS-SRTP is active)
            if self._dtls and self._dtls.encrypted:
                data = self._dtls.unprotect(data)
                if data is None:
                    continue  # decryption failed, drop

            if len(data) < RTP_HEADER_SIZE:
                continue

            # Parse RTP header — accept only our negotiated codec
            payload_type = data[1] & 0x7F
            if payload_type != self.codec.payload_type:
                continue

            payload = data[RTP_HEADER_SIZE:]
            # Decode wire → s16le, put in public rx_queue (for SIPSource)
            try:
                self.rx_queue.put_nowait(self.codec.decode(payload))
            except queue.Full:
                pass  # drop — real-time audio

    def _build_rtp_packet(self, payload: bytes) -> bytes:
        """Build RTP packet with negotiated codec payload type."""
        header = struct.pack("!BBHII",
                             0x80,
                             self.codec.payload_type,
                             self._tx_seq,
                             self._tx_ts,
                             self._tx_ssrc)
        return header + payload


class _RTPClient:
    """Minimal adapter so SIPSink can replace our output buffer."""

    def __init__(self, session: RTPSession):
        self._session = session
        self.pmout = None  # SIPSink replaces this with _AudioQueue
        self.outIP = session.remote_host
        self.outPort = session.remote_port


class RTPCallSession:
    """Adapter that wraps RTPSession to look like a pyVoIP SIPSession.

    Pass to SIPSource/SIPSink via bridge_to_call::

        rtp = RTPSession(local_port, remote_host, remote_port, codec=PCMA)
        rtp.start()
        session = RTPCallSession(rtp)
        # Now use session with SIPSource(session), SIPSink(session)
    """

    def __init__(self, rtp_session: RTPSession):
        self._rtp = rtp_session
        self.connected = threading.Event()
        self.hungup = threading.Event()
        self.connected.set()  # already connected
        self.rx_queue = rtp_session.rx_queue  # for SIPSource

    @property
    def call(self):
        return self._rtp

    def hangup(self) -> None:
        """Signal end-of-call for monitor threads and stop RTP."""
        self.hungup.set()
        self._rtp.stop()
