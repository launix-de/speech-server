from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator, List, Optional

from .base import AudioFormat, Stage
from .QueueSource import QueueSource

_LOGGER = logging.getLogger("audio-tee")

# Bounded queue size — allow sidechains to lag behind briefly without
# dropping audio immediately. Transcript/STT sidechains are not
# latency-critical, but dropping chunks hurts recognition quality.
# put_nowait() drops on full so the main pipeline never blocks.
_QUEUE_MAXSIZE = 750


class AudioTee(Stage):
    """Processor: pass-through that copies data to side-chain sinks.

    Every chunk from upstream is yielded unchanged to downstream AND
    pushed into registered side-chain queues. Side-chain sinks run in
    daemon threads so they don't block the main pipeline.

    Hot-pluggable: sidechains and mixer feeds can be added or removed
    while the stream is running. Zero outputs at start is supported.

    Backpressure: bounded queues (maxsize=200), ``put_nowait()`` drops
    on full with a warning. Main pipeline never blocks.
    """

    def __init__(self, sample_rate: int, encoding: str = "s16le") -> None:
        super().__init__()
        fmt = AudioFormat(sample_rate, encoding)
        self.input_format = fmt
        self.output_format = fmt
        self._lock = threading.Lock()
        self._sidechain_queues: List[queue.Queue] = []
        self._sidechain_sinks: List[Stage] = []
        self._mixer_queues: List[queue.Queue] = []
        self._threads: List[threading.Thread] = []
        self._streaming = False  # True while stream_pcm24k is active

    def add_sidechain(self, sink: Stage) -> QueueSource:
        """Register a sink as a side-chain consumer.

        Returns the QueueSource that feeds the sink (already piped).
        Can be called before or during streaming. If called during
        streaming, the sink thread is started immediately.
        """
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        src = QueueSource(q, self.output_format.sample_rate, self.output_format.encoding)
        src.pipe(sink)
        with self._lock:
            self._sidechain_queues.append(q)
            self._sidechain_sinks.append(sink)
            if self._streaming:
                t = threading.Thread(target=self._run_sink, args=(sink,), daemon=True)
                t.start()
                self._threads.append(t)
        return src

    def remove_sidechain(self, sink: Stage) -> None:
        """Remove a side-chain sink. Sends EOF to its queue.

        Can be called while the stream is running.
        """
        with self._lock:
            try:
                idx = self._sidechain_sinks.index(sink)
            except ValueError:
                _LOGGER.warning("AudioTee: sink not found for removal")
                return
            q = self._sidechain_queues.pop(idx)
            self._sidechain_sinks.pop(idx)
        # Send EOF outside the lock
        try:
            q.put(None, timeout=1.0)
        except Exception:
            pass

    def add_mixer_feed(self, mixer_queue: queue.Queue) -> None:
        """Register a raw queue to feed (for named mixers).

        Can be called before or during streaming.
        """
        with self._lock:
            self._mixer_queues.append(mixer_queue)

    def remove_mixer_feed(self, mixer_queue: queue.Queue) -> None:
        """Remove a mixer feed queue. Sends EOF sentinel.

        Can be called while the stream is running.
        """
        with self._lock:
            try:
                self._mixer_queues.remove(mixer_queue)
            except ValueError:
                _LOGGER.warning("AudioTee: mixer queue not found for removal")
                return
        # Send EOF outside the lock
        try:
            mixer_queue.put(None, timeout=1.0)
        except Exception:
            pass

    def stream_pcm24k(self) -> Iterator[bytes]:
        if not self.upstream:
            return

        with self._lock:
            self._streaming = True
            # Start threads for sidechains registered before streaming began
            for sink in self._sidechain_sinks:
                t = threading.Thread(target=self._run_sink, args=(sink,), daemon=True)
                t.start()
                self._threads.append(t)

        try:
            for chunk in self.upstream.stream_pcm24k():
                if self.cancelled:
                    break
                # Snapshot the queue lists under lock
                with self._lock:
                    queues = list(self._sidechain_queues) + list(self._mixer_queues)
                # Copy to all side-chain and mixer queues
                for q in queues:
                    try:
                        q.put_nowait(chunk)
                    except queue.Full:
                        _LOGGER.warning("AudioTee: queue full, dropping chunk")
                # Pass through to downstream
                yield chunk
        finally:
            with self._lock:
                self._streaming = False
                queues = list(self._sidechain_queues) + list(self._mixer_queues)
                threads = list(self._threads)
            # Send EOF sentinel to all queues
            for q in queues:
                try:
                    q.put(None, timeout=1.0)
                except Exception:
                    pass
            # Wait for side-chain threads to finish
            for t in threads:
                t.join(timeout=5.0)

    @staticmethod
    def _run_sink(sink: Stage) -> None:
        """Run a side-chain sink in a background thread.

        Walks downstream to find the terminal sink (has run()).
        If no terminal found, drains via stream_pcm24k (yield-based).
        """
        # Walk to terminal sink
        current = sink
        while current.downstream is not None:
            current = current.downstream
        try:
            if hasattr(current, 'run') and current is not sink:
                current.run()
            elif hasattr(sink, 'run'):
                sink.run()
            else:
                # No run() anywhere — drain via yield
                for _ in sink.stream_pcm24k():
                    pass
        except Exception as e:
            _LOGGER.warning("AudioTee side-chain error: %s", e)

    def cancel(self) -> None:
        super().cancel()
        with self._lock:
            queues = list(self._sidechain_queues) + list(self._mixer_queues)
            sinks = list(self._sidechain_sinks)
            threads = list(self._threads)
        for q in queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for sink in sinks:
            try:
                sink.cancel()
            except Exception:
                pass
        for t in threads:
            t.join(timeout=1.0)
