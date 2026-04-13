"""WebSocket session for the Fourier codec.

Manages a single bidirectional connection at ``/ws/socket/<id>``.
Handles the profile handshake, then runs RX (receive → decode → rx_queue)
and TX (tx_queue → encode → send) loops.

Sessions are kept in a global registry so that ``CodecSocketSource`` and
``CodecSocketSink`` stages (which may live in different pipelines /
threads) can locate them by ID.
"""
from __future__ import annotations

import json
import logging
import threading
from queue import Queue, Empty, Full
from typing import Dict, List, Optional

from . import fourier_codec as codec

_LOGGER = logging.getLogger("codec-socket")

_sessions: Dict[str, "CodecSocketSession"] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str) -> Optional["CodecSocketSession"]:
    return _sessions.get(session_id)


class CodecSocketSession:
    """One WebSocket ↔ pipeline connection.

    *server_profiles* is the ordered list of profiles the server is
    willing to use (best first).  During the handshake the server picks
    the highest-priority profile that both sides support.
    """

    def __init__(
        self,
        session_id: str,
        server_profiles: Optional[List[str]] = None,
    ) -> None:
        self.session_id = session_id
        self.server_profiles = server_profiles or list(codec.PROFILE_NAMES)
        self.negotiated_profile: Optional[str] = None
        self.rx_queue: Queue[bytes] = Queue(maxsize=500)
        self.tx_queue: Queue[bytes] = Queue(maxsize=500)
        self.connected = threading.Event()
        self.closed = threading.Event()
        self.last_rx_profile: Optional[str] = None  # tracks client's current profile
        self._ws = None
        with _sessions_lock:
            _sessions[session_id] = self

    # ------------------------------------------------------------------
    #  WebSocket handler (called from Flask route)
    # ------------------------------------------------------------------

    def handle_ws(self, ws) -> None:
        """Run one WS connection's lifetime against this session.

        The session itself is re-entrant: when the browser does a
        normal disconnect/reconnect cycle (very common on page load,
        network blips), the old WS ends here, rx/tx loops exit, and
        the next ``handle_ws`` call picks up with a new ws on the
        same session — queues and profile state are preserved.

        Full teardown only happens via :meth:`close` (pipeline ending
        or explicit webclient close).
        """
        if self.closed.is_set():
            _LOGGER.info("CodecSocket %s: already closed, rejecting ws",
                         self.session_id)
            return
        self._ws = ws
        # Reset per-ws transient state; keep queues so audio doesn't get
        # lost between reconnects.
        self.connected.clear()
        try:
            self._handshake(ws)
            self.connected.set()
            _LOGGER.info(
                "CodecSocket %s: connected, profile=%s",
                self.session_id, self.negotiated_profile,
            )

            # RX in a background thread; TX in the current thread.
            rx_thread = threading.Thread(
                target=self._rx_loop, args=(ws,), daemon=True,
            )
            rx_thread.start()
            self._tx_loop(ws)
            rx_thread.join(timeout=2)
        except Exception as e:
            _LOGGER.warning("CodecSocket %s error: %s", self.session_id, e)
        finally:
            # Drop the current ws reference but DON'T tear down the
            # session — a quick browser reconnect needs to find it.
            # ``close()`` is called by the pipeline when the conference
            # ends or by webclient teardown.
            self._ws = None

    # ------------------------------------------------------------------
    #  Handshake
    # ------------------------------------------------------------------

    def _handshake(self, ws) -> None:
        """Negotiate profile with the client.

        Expected client message:
            {"type": "hello", "profiles": ["high", "medium", "low"]}
        If no hello arrives within 5 s, default to the server's first
        profile (backward-compatible).
        """
        try:
            msg = ws.receive(timeout=5)
        except Exception:
            msg = None

        client_profiles: List[str] = []
        if isinstance(msg, str):
            try:
                obj = json.loads(msg)
                if obj.get("type") == "hello":
                    client_profiles = [
                        p for p in obj.get("profiles", [])
                        if p in codec.PROFILES
                    ]
            except Exception:
                pass

        if client_profiles:
            # Pick the first server profile that the client also supports
            chosen = None
            for sp in self.server_profiles:
                if sp in client_profiles:
                    chosen = sp
                    break
            if chosen is None:
                # Fallback: first client profile we support
                for cp in client_profiles:
                    if cp in self.server_profiles:
                        chosen = cp
                        break
            self.negotiated_profile = chosen or self.server_profiles[0]
        else:
            self.negotiated_profile = self.server_profiles[0]

        # Send hello response
        ws.send(json.dumps({
            "type": "hello",
            "profile": self.negotiated_profile,
            "session_id": self.session_id,
        }))

    # ------------------------------------------------------------------
    #  RX loop: receive encoded frames → decode → rx_queue
    # ------------------------------------------------------------------

    def _rx_loop(self, ws) -> None:
        rx_count = 0
        try:
            while not self.closed.is_set():
                try:
                    msg = ws.receive(timeout=1)
                except Exception as e:
                    if self.closed.is_set():
                        break
                    continue
                if msg is None:
                    _LOGGER.info("CodecSocket %s RX: received None (WS closed)", self.session_id)
                    break
                if isinstance(msg, str):
                    if msg == "__END__":
                        _LOGGER.info("CodecSocket %s RX: received __END__", self.session_id)
                        break
                    continue
                # Binary: encoded codec frame
                try:
                    samples, frame_profile = codec.decode_frame(msg)
                    if frame_profile:
                        self.last_rx_profile = frame_profile
                    pcm = codec.float32_to_pcm_s16le(samples)
                    self.rx_queue.put_nowait(pcm)
                    rx_count += 1
                except Full:
                    pass
                except Exception as e:
                    _LOGGER.debug("CodecSocket %s RX decode error: %s", self.session_id, e)
        except Exception as e:
            _LOGGER.info("CodecSocket %s RX loop exception: %s", self.session_id, e)
        finally:
            _LOGGER.info("CodecSocket %s RX loop ended after %d frames", self.session_id, rx_count)
            # Do NOT set ``closed`` here — this loop exits on browser
            # WS disconnect (normal reconnect pattern) and the session
            # must survive so the next ws_codec_socket call finds it.
            # ``closed`` is set explicitly by :meth:`close` when the
            # pipeline tears down.

    # ------------------------------------------------------------------
    #  TX loop: tx_queue → encode → send
    # ------------------------------------------------------------------

    def _tx_loop(self, ws) -> None:
        frame_bytes = codec.FRAME_SAMPLES * 2  # s16le
        buf = b""
        try:
            while not self.closed.is_set():
                try:
                    data = self.tx_queue.get(timeout=0.5)
                except Empty:
                    continue
                if data is None:
                    break
                buf += data
                while len(buf) >= frame_bytes:
                    chunk = buf[:frame_bytes]
                    buf = buf[frame_bytes:]
                    samples = codec.pcm_s16le_to_float32(chunk)
                    # Use client's current profile (follows per-frame switching)
                    tx_profile = self.last_rx_profile or self.negotiated_profile or "low"
                    encoded = codec.encode_frame(samples, tx_profile)
                    try:
                        ws.send(encoded)
                    except Exception:
                        # WS died (browser navigated / network blip).
                        # Return without closing the session — the next
                        # handle_ws call will pick up with a new ws.
                        return
            # Flush remaining (zero-pad)
            if buf:
                padded = buf + b"\x00" * (frame_bytes - len(buf))
                samples = codec.pcm_s16le_to_float32(padded)
                tx_profile = self.last_rx_profile or self.negotiated_profile or "low"
                encoded = codec.encode_frame(samples, tx_profile)
                try:
                    ws.send(encoded)
                except Exception:
                    pass
        except Exception as e:
            _LOGGER.debug("CodecSocket %s TX loop ended: %s", self.session_id, e)
        # No finally close — session must survive browser WS disconnect
        # so the reconnect finds it.  ``close()`` is only called by
        # the pipeline teardown or webclient.close_webclient_session.

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.closed.set()
        self.connected.set()
        try:
            self.tx_queue.put_nowait(None)
        except Exception:
            pass
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        with _sessions_lock:
            _sessions.pop(self.session_id, None)
