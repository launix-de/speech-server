from __future__ import annotations

import logging
from typing import Iterator

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-source")


class SIPSource(Stage):
    """Source stage: reads audio from a pyVoIP or RTPSession SIP call.

    Output: unsigned 8-bit PCM (u8) mono @ 8000 Hz.
    pyVoIP decodes µ-law → u8 internally. RTPSession does the same.
    pipe() auto-inserts converters to the mixer's s16le@48kHz.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.output_format = AudioFormat(8000, "u8")

    def stream_pcm24k(self) -> Iterator[bytes]:
        import time as _time

        if not self.session.connected.is_set():
            _LOGGER.info("SIPSource: waiting for call to connect...")
            self.session.connected.wait(timeout=30)

        if self.session.hungup.is_set():
            _LOGGER.warning("SIPSource: call already hung up")
            return

        _LOGGER.info("SIPSource: streaming audio from SIP call")
        call = self.session.call
        while not self.cancelled and not self.session.hungup.is_set():
            try:
                frame = call.read_audio(length=160, blocking=False)
                if frame:
                    yield frame
                _time.sleep(0.02)
            except Exception as e:
                if not self.cancelled:
                    _LOGGER.warning("SIPSource read error: %s", e)
                break

        _LOGGER.info("SIPSource: stream ended")
