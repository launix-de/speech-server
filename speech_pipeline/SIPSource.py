"""Source stage: reads audio from a SIP call (pyVoIP or RTPSession).

Same pattern as AudioSocketSource — blocks on session.rx_queue.
The session's RTP receiver fills the queue; this stage just yields.
pipe() auto-inserts converters to the mixer's s16le@48kHz.
"""
from __future__ import annotations

import logging
from queue import Empty
from typing import Iterator

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-source")


class SIPSource(Stage):
    """Yields audio from session.rx_queue.

    Same pattern as AudioSocketSource/CodecSocketSource.
    Output format depends on what the session puts in rx_queue.
    """

    def __init__(self, session, leg=None) -> None:
        super().__init__()
        self.session = session
        self.leg = leg
        from speech_pipeline.RTPSession import RTPSession
        call = session.call if hasattr(session, 'call') else None
        if isinstance(call, RTPSession):
            self.output_format = AudioFormat(call.codec.sample_rate, "s16le")
        else:
            # pyVoIP path: PyVoIPCallSession._rx_pump decodes u-law → s16le
            # before queueing, so the data in rx_queue is always s16le.
            self.output_format = AudioFormat(8000, "s16le")

    def _on_close(self) -> None:
        """Orderly teardown: send SIP BYE/CANCEL (idempotent)."""
        if self.session.hungup.is_set():
            return
        if self.leg is not None:
            try:
                self.leg.hangup()
                return
            except Exception as e:
                _LOGGER.debug("SIPSource: leg.hangup raised: %s", e)
        try:
            self.session.hangup()
        except Exception as e:
            _LOGGER.debug("SIPSource: session.hangup raised: %s", e)

    def stream_pcm24k(self) -> Iterator[bytes]:
        if not self.session.connected.is_set():
            _LOGGER.info("SIPSource: waiting for call to connect...")
            self.session.connected.wait(timeout=30)

        if self.session.hungup.is_set():
            _LOGGER.warning("SIPSource: call already hung up")
            return

        _LOGGER.info("SIPSource: streaming audio (%s)", self.output_format)
        natural_eof = False
        try:
            while not self.cancelled and not self.session.hungup.is_set():
                try:
                    frame = self.session.rx_queue.get(timeout=0.5)
                except Empty:
                    continue
                if frame is None:
                    natural_eof = True
                    break
                yield frame
            # Remote BYE (session.hungup) also counts as natural EOF.
            if self.session.hungup.is_set():
                natural_eof = True
        finally:
            _LOGGER.info("SIPSource: stream ended (eof=%s, cancelled=%s)",
                         natural_eof, self.cancelled)
            if natural_eof and not self.cancelled:
                self.close()
