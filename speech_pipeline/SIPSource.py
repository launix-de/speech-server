"""Source stage: reads audio from a SIP call (pyVoIP or RTPSession).

Same pattern as CodecSocketSource — blocks on session.rx_queue,
yields s16le @ 48 kHz. The session handles decoding + upsampling
so pipe() inserts NO SampleRateConverter (same as websocket codec).
"""
from __future__ import annotations

import logging
from queue import Empty
from typing import Iterator

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-source")


class SIPSource(Stage):
    """Yields PCM s16le mono 48 kHz from a SIP session's rx_queue.

    Identical to CodecSocketSource — the session does all format
    conversion so the pipeline is converter-free to the mixer.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.output_format = AudioFormat(48000, "s16le")

    def stream_pcm24k(self) -> Iterator[bytes]:
        if not self.session.connected.is_set():
            _LOGGER.info("SIPSource: waiting for call to connect...")
            self.session.connected.wait(timeout=30)

        if self.session.hungup.is_set():
            _LOGGER.warning("SIPSource: call already hung up")
            return

        _LOGGER.info("SIPSource: streaming audio (48kHz s16le)")
        while not self.cancelled and not self.session.hungup.is_set():
            try:
                frame = self.session.rx_queue.get(timeout=0.5)
            except Empty:
                continue
            if frame:
                yield frame

        _LOGGER.info("SIPSource: stream ended")
