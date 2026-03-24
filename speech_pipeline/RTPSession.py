"""Minimal RTP session — provides the same audio interface as pyVoIP calls.

Used by SIPSource/SIPSink to bridge sip_stack-originated calls into
the Stage pipeline.  Handles G.711 µ-law (PCMU) over raw UDP sockets.

Interface contract (compatible with pyVoIP VoIPCall):
- ``read_audio(length, blocking)`` → PCM u8 bytes
- ``write_audio(data)`` → send PCM u8 bytes
- ``RTPClients[0].pmout`` → output buffer (replaced by SIPSink)
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

_LOGGER = logging.getLogger("rtp-session")

# RTP constants
RTP_HEADER_SIZE = 12
PCMU_PAYLOAD_TYPE = 0
FRAME_SIZE = 160        # 20ms @ 8000 Hz
SAMPLE_RATE = 8000


class RTPSession:
    """Raw UDP RTP session that mimics pyVoIP's call interface.

    After construction, wrap in a ``_CallSession``-like adapter and
    pass to ``SIPSource``/``SIPSink`` — they work identically to pyVoIP.
    """

    def __init__(self, local_port: int, remote_host: str, remote_port: int):
        self.local_port = local_port
        self.remote_host = remote_host
        self.remote_port = remote_port

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", local_port))
        self._sock.settimeout(0.1)

        self._rx_queue: queue.Queue = queue.Queue(maxsize=100)
        self._tx_queue: queue.Queue = queue.Queue()
        self._running = False
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None

        # TX state
        self._tx_seq = 0
        self._tx_ts = 0
        self._tx_ssrc = struct.unpack("!I", struct.pack("!I", id(self) & 0xFFFFFFFF))[0]

        # Fake RTPClient for SIPSink compatibility
        self._rtp_client = _RTPClient(self)
        self.RTPClients = [self._rtp_client]

        self._tx_packet_count = 0
        _LOGGER.info("RTPSession: %s:%d ↔ %s:%d",
                      "0.0.0.0", local_port, remote_host, remote_port)

    def start(self) -> None:
        """Start RTP receive and transmit threads."""
        self._running = True
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
            self._sock.close()
        except Exception:
            pass

    def read_audio(self, length: int = FRAME_SIZE, blocking: bool = False) -> bytes:
        """Read audio from RTP, decoded to linear u8 (compatible with SIPSource).

        RTP payload is µ-law (PCMU). Decodes to linear unsigned 8-bit PCM
        (silence=128) matching the pipeline's u8 format.
        """
        try:
            timeout = 0.02 if blocking else 0.0
            ulaw_data = self._rx_queue.get(timeout=timeout)
        except queue.Empty:
            return b""
        if not ulaw_data:
            return b""
        # µ-law → u8 (same as pyVoIP's parse_pcmu)
        data = audioop.ulaw2lin(ulaw_data, 1)
        return audioop.bias(data, 1, 128)

    def write_audio(self, data: bytes) -> None:
        """Buffer u8 audio for paced RTP transmission.

        Receives u8 PCM (same as pyVoIP), encodes to µ-law for RTP.
        """
        # u8 → µ-law (same as pyVoIP's encode_pcmu)
        ulaw_data = audioop.bias(data, 1, -128)
        ulaw_data = audioop.lin2ulaw(ulaw_data, 1)
        self._tx_queue.put(ulaw_data)

    def _tx_loop(self) -> None:
        """Send RTP packets at exactly 20ms intervals (160 bytes PCMU)."""
        buf = b""
        next_send = time.monotonic()

        while self._running:
            # Collect data from queue
            try:
                while True:
                    buf += self._tx_queue.get_nowait()
            except queue.Empty:
                pass

            # Send one frame if we have enough data
            if len(buf) >= FRAME_SIZE:
                chunk = buf[:FRAME_SIZE]
                buf = buf[FRAME_SIZE:]

                packet = self._build_rtp_packet(chunk)
                try:
                    self._sock.sendto(packet, (self.remote_host, self.remote_port))
                except Exception:
                    pass
                self._tx_seq = (self._tx_seq + 1) & 0xFFFF
                self._tx_ts += FRAME_SIZE
            else:
                # Not enough data — send silence
                packet = self._build_rtp_packet(b"\xff" * FRAME_SIZE)
                try:
                    self._sock.sendto(packet, (self.remote_host, self.remote_port))
                except Exception:
                    pass
                self._tx_seq = (self._tx_seq + 1) & 0xFFFF
                self._tx_ts += FRAME_SIZE

            # Pace at exactly 20ms
            next_send += 0.020
            sleep_time = next_send - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_send = time.monotonic()  # catch up

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

            if len(data) < RTP_HEADER_SIZE:
                continue

            # Parse RTP header
            payload_type = data[1] & 0x7F
            if payload_type != PCMU_PAYLOAD_TYPE:
                continue  # skip non-PCMU (e.g. DTMF events)

            payload = data[RTP_HEADER_SIZE:]
            try:
                self._rx_queue.put_nowait(payload)
            except queue.Full:
                pass  # drop oldest — real-time audio

    def _build_rtp_packet(self, payload: bytes) -> bytes:
        """Build a minimal RTP packet with PCMU payload."""
        # V=2, P=0, X=0, CC=0, M=0, PT=0 (PCMU)
        header = struct.pack("!BBHII",
                             0x80,                    # V=2
                             PCMU_PAYLOAD_TYPE,       # PT=0 (PCMU)
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

        rtp = RTPSession(local_port, remote_host, remote_port)
        rtp.start()
        session = RTPCallSession(rtp)
        # Now use session with SIPSource(session), SIPSink(session)
    """

    def __init__(self, rtp_session: RTPSession):
        self._rtp = rtp_session
        self.connected = threading.Event()
        self.hungup = threading.Event()
        self.connected.set()  # already connected

    @property
    def call(self):
        return self._rtp
