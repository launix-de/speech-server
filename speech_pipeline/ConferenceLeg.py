"""Bidirectional conference participant stage.

A single Stage that connects upstream (mic/codec) and downstream
(speaker/codec) to a ConferenceMixer with automatic mix-minus.

    codec:X | conference:CALL | codec:X

The ConferenceLeg:
- Passes itself as a source Stage to the mixer (yield-based, no pump thread)
- Receives conference mix (minus own input) → yields to downstream

All format conversion happens via ``pipe()`` auto-resampling.
No manual queues, no MixMinus stage, no AudioTee — the
ConferenceMixer handles mix-minus atomically.
"""
from __future__ import annotations

import audioop
import logging
import queue
from typing import Iterator, Optional

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("conference-leg")


class ConferenceLeg(Stage):
    """Processor: bidirectional conference participant.

    Upstream provides this participant's audio (e.g. SIPSource via SRC).
    Downstream receives the conference mix minus own audio.

    Must be connected to a ConferenceMixer via ``attach(mixer)``.
    The mixer's own pump thread drives the upstream generator — no
    extra pump thread needed.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        fmt = AudioFormat(sample_rate, "s16le")
        self.input_format = fmt
        self.output_format = fmt
        self._mixer = None
        self._src_id: Optional[str] = None
        self._sink_id: Optional[str] = None
        self._out_q: Optional[queue.Queue] = None
        self.on_attached = None   # callable, fired when leg is live
        self.on_detached = None   # callable, fired when leg disconnects

    def attach(self, mixer) -> None:
        """Attach to a ConferenceMixer.  Called by the DSL builder."""
        self._mixer = mixer

    def stream_pcm24k(self) -> Iterator[bytes]:
        if not self.upstream or not self._mixer:
            return

        # Register as full participant — mixer drives our upstream via
        # its own pump thread (add_source inside add_participant).
        self._src_id, self._sink_id, self._out_q = \
            self._mixer.add_participant(self.upstream)

        _LOGGER.info("ConferenceLeg: attached (src=%s)", self._src_id)

        if self.on_attached:
            try:
                self.on_attached(self)
            except Exception as e:
                _LOGGER.warning("ConferenceLeg on_attached error: %s", e)

        # Yield conference output to downstream
        frame_count = 0
        try:
            while not self.cancelled:
                try:
                    frame = self._out_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                frame_count += 1
                if frame_count == 1 or frame_count % 500 == 0:
                    _LOGGER.info("ConferenceLeg %s: yielded %d frames",
                                 self._src_id, frame_count)
                yield frame
        finally:
            if self._src_id:
                self._mixer.kill_source(self._src_id)
            if self._sink_id:
                self._mixer.remove_sink(self._sink_id)
            _LOGGER.info("ConferenceLeg: detached (src=%s)", self._src_id)
            if self.on_detached:
                try:
                    self.on_detached(self)
                except Exception as e:
                    _LOGGER.warning("ConferenceLeg on_detached error: %s", e)

    def cancel(self) -> None:
        super().cancel()
        if self._out_q:
            try:
                self._out_q.put_nowait(None)
            except Exception:
                pass
