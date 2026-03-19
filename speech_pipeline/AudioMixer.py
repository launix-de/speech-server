from __future__ import annotations

import audioop
import logging
import queue
import threading
import time
from typing import Iterator, List

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("audio-mixer")


class AudioMixer(Stage):
    """Source stage: mixes N input queues into a single PCM output.

    Each input is a ``queue.Queue[bytes | None]`` fed by an AudioTee
    (via ``add_mixer_feed()``) or directly by application code.

    Mixing uses ``audioop.add()`` on fixed-size frames (default 20ms).
    Sources finishing at different times contribute silence.
    The mixer continues until ALL inputs have sent the ``None`` sentinel.

    Hot-pluggable: inputs can be added or removed while the stream is
    running. Zero inputs at start is supported — the mixer will wait
    until at least one input is added before producing output.
    """

    def __init__(self, name: str, sample_rate: int = 16000, frame_ms: int = 20) -> None:
        super().__init__()
        self.name = name
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # s16le = 2 bytes/sample
        self.output_format = AudioFormat(sample_rate, "s16le")
        self._lock = threading.Lock()
        self._inputs: List[queue.Queue] = []
        self._buffers: List[bytearray] = []
        self._finished: List[bool] = []
        # Event signalled when first input is added (unblocks stream_pcm24k)
        self._has_inputs = threading.Event()

    def add_input(self) -> queue.Queue:
        """Register an input source. Returns queue to push PCM into.

        Push ``bytes`` chunks to feed audio. Push ``None`` to signal EOF.
        Can be called before or during streaming.
        """
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._inputs.append(q)
            self._buffers.append(bytearray())
            self._finished.append(False)
        self._has_inputs.set()
        return q

    def remove_input(self, q: queue.Queue) -> None:
        """Remove an input by its queue reference.

        Can be called while the stream is running. The input's buffered
        data is discarded.
        """
        with self._lock:
            try:
                idx = self._inputs.index(q)
            except ValueError:
                _LOGGER.warning("AudioMixer '%s': input queue not found for removal", self.name)
                return
            self._inputs.pop(idx)
            self._buffers.pop(idx)
            self._finished.pop(idx)
            _LOGGER.debug("AudioMixer '%s': removed input, %d remaining", self.name, len(self._inputs))

    def stream_pcm24k(self) -> Iterator[bytes]:
        if hasattr(self, '_streaming') and self._streaming:
            raise RuntimeError(
                f"AudioMixer '{self.name}' is already being consumed. "
                f"Use AudioTee to distribute to multiple consumers."
            )
        self._streaming = True
        # Wait for at least one input (or cancellation)
        while not self.cancelled:
            if self._has_inputs.wait(timeout=0.5):
                break
        if self.cancelled:
            return

        _LOGGER.info("AudioMixer '%s': starting @ %d Hz, %d ms frames",
                      self.name, self.sample_rate, self.frame_ms)

        silence = b"\x00" * self.frame_bytes

        while not self.cancelled:
            with self._lock:
                n = len(self._inputs)
                if n == 0:
                    # All inputs removed — sleep and retry
                    pass
                else:
                    # Drain all input queues into per-source buffers
                    for i in range(n):
                        if self._finished[i]:
                            continue
                        while True:
                            try:
                                chunk = self._inputs[i].get_nowait()
                            except queue.Empty:
                                break
                            if chunk is None:
                                self._finished[i] = True
                                _LOGGER.debug("AudioMixer '%s': input %d finished", self.name, i)
                                break
                            self._buffers[i].extend(chunk)

                    # All current inputs finished and all buffers drained?
                    all_done = n > 0 and all(self._finished)
                    remaining = sum(len(b) for b in self._buffers) if all_done else 0

            if n == 0:
                time.sleep(self.frame_ms / 1000.0)
                continue

            if all_done and remaining < self.frame_bytes:
                break

            # Check if we have at least one frame from any source
            with self._lock:
                has_data = any(len(b) >= self.frame_bytes for b in self._buffers)
                check_all_done = all(self._finished) if self._finished else False

            if not has_data and not check_all_done:
                time.sleep(self.frame_ms / 1000.0)
                continue

            # Extract one frame from each buffer, mix together
            with self._lock:
                mixed = silence
                for i, buf in enumerate(self._buffers):
                    if len(buf) >= self.frame_bytes:
                        frame = bytes(buf[:self.frame_bytes])
                        del buf[:self.frame_bytes]
                    else:
                        frame = silence
                    mixed = audioop.add(mixed, frame, 2)

            yield mixed

        _LOGGER.info("AudioMixer '%s': done", self.name)
