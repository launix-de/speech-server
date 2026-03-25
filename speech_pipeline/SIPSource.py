from __future__ import annotations

import logging
from typing import Iterator

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-source")


class SIPSource(Stage):
    """Source stage: reads audio from a pyVoIP or RTPSession SIP call.

    For RTPSession: output is s16le @ codec sample rate (codec decodes).
    For pyVoIP: output is u8 @ 8000 Hz (pyVoIP decodes internally).
    pipe() auto-inserts converters to the mixer's s16le@48kHz.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        # Detect output format based on call type
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

        _LOGGER.info("SIPSource: streaming audio (%s)", self.output_format)
        call = self.session.call
        while not self.cancelled and not self.session.hungup.is_set():
            try:
                frame = call.read_audio(length=160, blocking=True)
                if frame:
                    yield frame
            except Exception as e:
                if not self.cancelled:
                    _LOGGER.warning("SIPSource read error: %s", e)
                break

        _LOGGER.info("SIPSource: stream ended")
