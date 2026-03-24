from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-source")


class SIPSource(Stage):
    """Source stage: reads audio from a SIP call (pyVoIP or RTPSession).

    Uses a bounded queue with drop-oldest to prevent drift buildup.
    Same code path for all backends — only output_format differs.

    pipe() auto-inserts converters to the mixer's s16le@48kHz.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        from speech_pipeline.RTPSession import RTPSession
        call = session.call if hasattr(session, 'call') else None
        if isinstance(call, RTPSession):
            self.output_format = AudioFormat(call.codec.sample_rate, "s16le")
        else:
            self.output_format = AudioFormat(8000, "u8")

    def stream_pcm24k(self) -> Iterator[bytes]:
        import time as _time

        if not self.session.connected.is_set():
            _LOGGER.info("SIPSource: waiting for call to connect...")
            self.session.connected.wait(timeout=30)

        if self.session.hungup.is_set():
            _LOGGER.warning("SIPSource: call already hung up")
            return

        call = self.session.call
        _LOGGER.info("SIPSource: streaming audio (%s)", self.output_format)

        # Bounded queue — drop oldest on overflow. Prevents drift buildup
        # regardless of backend timing (pyVoIP, RTPSession, or anything else).
        rx_q: queue.Queue = queue.Queue(maxsize=5)

        def _pump():
            while not self.cancelled and not self.session.hungup.is_set():
                try:
                    frame = call.read_audio(length=160, blocking=True)
                    if not frame:
                        continue
                    try:
                        rx_q.put_nowait(frame)
                    except queue.Full:
                        try:
                            rx_q.get_nowait()
                        except queue.Empty:
                            pass
                        rx_q.put_nowait(frame)
                except Exception:
                    if not self.cancelled:
                        break

        threading.Thread(target=_pump, daemon=True, name="sip-rx-pump").start()

        while not self.cancelled and not self.session.hungup.is_set():
            try:
                got_any = False
                for _ in range(10):
                    try:
                        frame = rx_q.get_nowait()
                    except queue.Empty:
                        break
                    yield frame
                    got_any = True
                if not got_any:
                    _time.sleep(0.02)
            except Exception as e:
                if not self.cancelled:
                    _LOGGER.warning("SIPSource read error: %s", e)
                break

        _LOGGER.info("SIPSource: stream ended")
