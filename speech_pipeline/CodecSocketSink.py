"""Sink stage: encodes upstream PCM to codec frames and sends via
CodecSocketSession's tx_queue."""
from __future__ import annotations

import logging
from typing import Iterator

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("codec-sink")


class CodecSocketSink(Stage):
    """Terminal sink — drives the pipeline like ``AudioSocketSink.run()``.

    Reads s16le mono 48 kHz PCM from the upstream stage, pushes it
    into the session's tx_queue where the TX loop encodes and sends it.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.input_format = AudioFormat(48000, "s16le")

    def run(self) -> None:
        if not self.session.connected.is_set():
            _LOGGER.info("CodecSocketSink: waiting for connection...")
            self.session.connected.wait(timeout=60)

        if self.session.closed.is_set():
            _LOGGER.warning("CodecSocketSink: already closed")
            return

        _LOGGER.info("CodecSocketSink: streaming to codec socket")
        from queue import Full as _QFull
        try:
            for pcm in self.upstream.stream_pcm24k():
                if self.cancelled or self.session.closed.is_set():
                    break
                # Non-blocking: if the queue is full (browser is between
                # reconnects, TX loop paused), drop the frame instead of
                # stalling the sink — real-time audio, better to lose a
                # few frames than to tear down the whole session.
                try:
                    self.session.tx_queue.put_nowait(pcm)
                except _QFull:
                    # Drain one old frame to keep the queue fresh.
                    try:
                        self.session.tx_queue.get_nowait()
                        self.session.tx_queue.put_nowait(pcm)
                    except Exception:
                        pass
        except Exception as e:
            _LOGGER.warning("CodecSocketSink error: %s", e)
        finally:
            _LOGGER.info("CodecSocketSink: stream ended")
            try:
                self.session.tx_queue.put_nowait(None)
            except Exception:
                pass
