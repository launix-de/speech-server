"""Sink stage: pushes PCM chunks into a ``queue.Queue``.

Counterpart to ``QueueSource``.  Used to bridge a Stage pipeline
into an AudioMixer input or any other queue-based consumer.
"""
from __future__ import annotations

import logging
import queue
from typing import Iterator

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("queue-sink")


class QueueSink(Stage):
    """Sink: reads from upstream and pushes into a queue."""

    def __init__(self, q: queue.Queue, sample_rate: int,
                 encoding: str = "s16le") -> None:
        super().__init__()
        self.q = q
        self.input_format = AudioFormat(sample_rate, encoding)

    def run(self) -> None:
        """Drive the pipeline — iterate upstream, push into queue."""
        if not self.upstream:
            return
        try:
            for chunk in self.upstream.stream_pcm24k():
                if self.cancelled:
                    break
                try:
                    self.q.put(chunk, timeout=1)
                except queue.Full:
                    _LOGGER.warning("QueueSink: queue full, dropping")
        finally:
            try:
                self.q.put(None, timeout=1)  # EOF sentinel
            except queue.Full:
                pass
