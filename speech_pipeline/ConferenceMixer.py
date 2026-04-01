"""Conference mixer with built-in atomic mix-minus.

A Stage that mixes N sources and distributes to M sinks with
per-sink echo cancellation.  Mix-minus is computed in the same
frame tick — no external sync needed.

All connections use the Stage ``pipe()`` interface for automatic
format conversion (sample rate, encoding).

Usage::

    conf = ConferenceMixer("my-conf", sample_rate=48000)

    # Full participant (speaks + listens without self-echo):
    src_id = conf.add_source(sip_source_stage)
    conf.add_sink(sip_sink_stage, mute_source=src_id)

    # Producer only (TTS):
    conf.add_source(tts_stage)

    # Consumer only (STT):
    conf.add_sink(whisper_stage)

    # Run (blocks — use a thread):
    conf.run()
"""
from __future__ import annotations

import audioop
import logging
import queue
import threading
import time
from typing import Dict, List, Optional

from .base import AudioFormat, Stage
from .QueueSink import QueueSink
from .QueueSource import QueueSource

_LOGGER = logging.getLogger("conference-mixer")

_ID_COUNTER = 0
_ID_LOCK = threading.Lock()

def _next_id(prefix: str) -> str:
    global _ID_COUNTER
    with _ID_LOCK:
        _ID_COUNTER += 1
        return f"{prefix}-{_ID_COUNTER}"


class ConferenceMixer(Stage):
    """Stage: N-to-M mixer with per-sink mix-minus.

    Inherits from Stage so it fits in the pipeline framework, but
    manages its own sources/sinks internally.
    """

    def __init__(self, name: str, sample_rate: int = 48000,
                 frame_ms: int = 20, frame_samples: int = 0) -> None:
        super().__init__()
        self.name = name
        self.sample_rate = sample_rate
        if frame_samples > 0:
            self.frame_samples = frame_samples
        else:
            self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_ms = self.frame_samples * 1000.0 / sample_rate
        self.frame_bytes = self.frame_samples * 2  # s16le
        self.output_format = AudioFormat(sample_rate, "s16le")

        self._lock = threading.Lock()
        self._sources: Dict[str, _Source] = {}
        self._sinks: List[_Sink] = []
        self._has_sources = threading.Event()
        self._running = False

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def add_source(self, stage: Stage) -> str:
        """Connect a source Stage to the conference.

        The stage's output is piped (with auto-conversion) into the
        mixer.  Returns a source ID for use with ``mute_source``.
        """
        src_id = _next_id("src")
        in_q: queue.Queue = queue.Queue(maxsize=200)
        sink = QueueSink(in_q, self.sample_rate, "s16le")
        stage.pipe(sink)

        entry = _Source(id=src_id, queue=in_q, sink=sink)

        # Run the source→QueueSink pipeline in a thread
        def _pump():
            try:
                sink.run()
            except Exception as e:
                _LOGGER.warning("ConferenceMixer '%s' source %s error: %s",
                                self.name, src_id, e)
            finally:
                entry.finished = True
                # Don't set done yet — mixer timer will set it
                # when buffer is drained (so audio is actually played)

        entry.thread = threading.Thread(target=_pump, daemon=True,
                                        name=f"csrc-{src_id}")
        with self._lock:
            self._sources[src_id] = entry
        self._has_sources.set()
        entry.thread.start()
        _LOGGER.info("ConferenceMixer '%s': +source %s", self.name, src_id)
        return src_id

    def add_input(self) -> queue.Queue:
        """Compat with AudioMixer: add a raw queue-based input.

        Returns a queue to push s16le PCM into (same as AudioMixer).
        Used by PipelineBuilder's ``tee:NAME`` element.
        """
        src_id = _next_id("src")
        in_q: queue.Queue = queue.Queue(maxsize=200)
        entry = _Source(id=src_id, queue=in_q, sink=None)
        with self._lock:
            self._sources[src_id] = entry
        self._has_sources.set()
        _LOGGER.info("ConferenceMixer '%s': +input %s", self.name, src_id)
        return in_q

    def remove_input(self, q: queue.Queue) -> None:
        """Compat with AudioMixer: remove by queue reference."""
        with self._lock:
            to_remove = [sid for sid, s in self._sources.items() if s.queue is q]
            for sid in to_remove:
                src = self._sources.pop(sid)
                src.buffer.clear()

    def add_participant(self, stage: Stage) -> tuple:
        """Add a full participant (input + auto-muted output).

        *stage* is the upstream audio source (e.g. a ConferenceLeg that
        wraps SIPSource).  The mixer's own pump thread drives the
        generator — no external pump needed.

        Returns ``(src_id, sink_id, out_queue)``.  The output automatically
        subtracts this participant's own input (mix-minus).
        """
        src_id = self.add_source(stage)
        sink_id = _next_id("out")
        out_q: queue.Queue = queue.Queue(maxsize=200)
        sink_entry = _Sink(id=sink_id, stage=None, queue=out_q,
                           source=None, mute_source=src_id, thread=None)
        with self._lock:
            self._sinks.append(sink_entry)
        _LOGGER.info("ConferenceMixer '%s': +participant %s", self.name, src_id)
        return src_id, sink_id, out_q

    def wait_source(self, src_id: str, timeout: float = None) -> bool:
        """Block until a source is fully consumed (buffer drained).

        Returns True if done, False on timeout.
        """
        with self._lock:
            entry = self._sources.get(src_id)
        if not entry:
            return True
        return entry.done.wait(timeout=timeout)

    def kill_source(self, src_id: str) -> None:
        """Immediately stop a source: cancel pipeline, clear buffer, remove."""
        with self._lock:
            entry = self._sources.pop(src_id, None)
        if entry:
            if entry.sink:
                entry.sink.cancel()
            entry.buffer.clear()
            entry.finished = True
            entry.done.set()
            _LOGGER.info("ConferenceMixer '%s': killed source %s",
                         self.name, src_id)

    def remove_source(self, src_id: str) -> None:
        """Disconnect a source."""
        with self._lock:
            entry = self._sources.pop(src_id, None)
        if entry:
            entry.sink.cancel()
            _LOGGER.info("ConferenceMixer '%s': -source %s",
                         self.name, src_id)

    # ------------------------------------------------------------------
    # Sinks
    # ------------------------------------------------------------------

    def add_sink(self, stage: Stage,
                 mute_source: Optional[str] = None) -> str:
        """Connect a sink Stage to the conference.

        The conference mix is piped (with auto-conversion) into the
        stage.  If *mute_source* is set, that source is subtracted
        from the mix (mix-minus / echo cancel).

        Returns a sink ID.
        """
        sink_id = _next_id("sink")
        out_q: queue.Queue = queue.Queue(maxsize=200)
        src = QueueSource(out_q, self.sample_rate, "s16le")
        src.pipe(stage)

        entry = _Sink(id=sink_id, stage=stage, queue=out_q,
                      source=src, mute_source=mute_source, thread=None)

        with self._lock:
            self._sinks.append(entry)
            if self._running:
                entry.thread = self._start_sink_thread(entry)

        _LOGGER.info("ConferenceMixer '%s': +sink %s (mute=%s)",
                     self.name, sink_id, mute_source)
        return sink_id

    def add_output(self, mute_source: Optional[str] = None) -> queue.Queue:
        """Compat with AudioMixer: add a raw queue-based output.

        Returns a queue to read s16le PCM from.
        Used by PipelineBuilder's ``mix:NAME`` element.
        """
        sink_id = _next_id("out")
        out_q: queue.Queue = queue.Queue(maxsize=200)
        entry = _Sink(id=sink_id, stage=None, queue=out_q,
                      source=None, mute_source=mute_source, thread=None)
        with self._lock:
            self._sinks.append(entry)
        _LOGGER.info("ConferenceMixer '%s': +output %s (mute=%s)",
                     self.name, sink_id, mute_source)
        return out_q

    def remove_sink(self, sink_id: str) -> None:
        """Disconnect a sink."""
        with self._lock:
            for i, entry in enumerate(self._sinks):
                if entry.id == sink_id:
                    self._sinks.pop(i)
                    if entry.source is not None:
                        entry.source.cancel()
                    try:
                        entry.queue.put_nowait(None)
                    except queue.Full:
                        pass
                    _LOGGER.info("ConferenceMixer '%s': -sink %s",
                                 self.name, sink_id)
                    return

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def stream_pcm24k(self):
        """Yield conference mix from an internal output queue.

        Used by PipelineBuilder's ``mix:NAME`` when the ConferenceMixer
        is set as a named mixer.  Automatically creates an output queue.
        """
        out_q = self.add_output()
        while not self.cancelled:
            try:
                frame = out_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            yield frame

    def run(self) -> None:
        """Main loop — blocks until cancelled."""
        self._running = True

        while not self.cancelled:
            if self._has_sources.wait(timeout=0.5):
                break
        if self.cancelled:
            return

        _LOGGER.info("ConferenceMixer '%s': started @ %d Hz, %d ms",
                     self.name, self.sample_rate, self.frame_ms)

        # Start sink threads
        with self._lock:
            for entry in self._sinks:
                if entry.thread is None and entry.stage is not None:
                    entry.thread = self._start_sink_thread(entry)

        silence = b"\x00" * self.frame_bytes
        frame_interval = self.frame_ms / 1000.0
        next_tick = time.monotonic()

        while not self.cancelled:
            # Real-time pacing: one frame per frame_interval
            now = time.monotonic()
            wait = next_tick - now
            if wait > 0:
                time.sleep(wait)
            next_tick += frame_interval

            with self._lock:
                sources = dict(self._sources)

            if not sources:
                continue

            # Step 1: drain queues, extract one frame per source
            frames: Dict[str, bytes] = {}
            for sid, src in sources.items():
                while True:
                    try:
                        chunk = src.queue.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:
                        src.finished = True
                        break
                    src.buffer.extend(chunk)

                if len(src.buffer) >= self.frame_bytes:
                    frames[sid] = bytes(src.buffer[:self.frame_bytes])
                    del src.buffer[:self.frame_bytes]
                elif src.finished and src.buffer:
                    # Flush the final short tail once instead of dropping it.
                    frames[sid] = bytes(src.buffer) + (b"\x00" * (self.frame_bytes - len(src.buffer)))
                    src.buffer.clear()
                else:
                    frames[sid] = silence
                    if src.finished:
                        src.done.set()

            # Step 2: full mix
            full_mix = silence
            for frame in frames.values():
                full_mix = audioop.add(full_mix, frame, 2)

            # Step 3: per-sink mix-minus and distribute
            with self._lock:
                sinks = list(self._sinks)

            for entry in sinks:
                if entry.mute_source and entry.mute_source in frames:
                    negated = audioop.mul(frames[entry.mute_source], 2, -1)
                    out = audioop.add(full_mix, negated, 2)
                else:
                    out = full_mix
                try:
                    entry.queue.put_nowait(out)
                except queue.Full:
                    pass

        # EOF to all sinks
        with self._lock:
            for entry in self._sinks:
                try:
                    entry.queue.put_nowait(None)
                except queue.Full:
                    pass

        self._running = False
        _LOGGER.info("ConferenceMixer '%s': done", self.name)

    def cancel(self) -> None:
        self.cancelled = True
        self._has_sources.set()
        with self._lock:
            for src in self._sources.values():
                if src.sink:
                    src.sink.cancel()
            for entry in self._sinks:
                if entry.source:
                    entry.source.cancel()
        super().cancel()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _find_terminal_sink(stage: Stage) -> Stage:
        """Walk downstream to find the terminal sink (has run())."""
        current = stage
        while current.downstream is not None:
            current = current.downstream
        return current

    def _start_sink_thread(self, entry: "_Sink") -> threading.Thread:
        def _run():
            try:
                # Find the terminal sink in the chain and run it
                terminal = self._find_terminal_sink(entry.stage)
                if hasattr(terminal, 'run'):
                    terminal.run()
                else:
                    # No terminal sink — drain the last stage
                    for _ in terminal.stream_pcm24k():
                        if self.cancelled:
                            break
            except Exception as e:
                _LOGGER.warning("ConferenceMixer '%s' sink error: %s",
                                self.name, e)

        t = threading.Thread(target=_run, daemon=True,
                             name=f"csink-{entry.id}")
        t.start()
        return t


class _Source:
    __slots__ = ("id", "queue", "sink", "buffer", "finished", "done", "thread")

    def __init__(self, id: str, queue: queue.Queue, sink: Optional[QueueSink]):
        self.id = id
        self.queue = queue
        self.sink = sink
        self.buffer = bytearray()
        self.finished = False
        self.done = threading.Event()  # set when finished AND buffer drained
        self.thread: Optional[threading.Thread] = None


class _Sink:
    __slots__ = ("id", "stage", "queue", "source", "mute_source", "thread")

    def __init__(self, id: str, stage: Stage, queue: queue.Queue,
                 source: QueueSource, mute_source: Optional[str],
                 thread: Optional[threading.Thread]):
        self.id = id
        self.stage = stage
        self.queue = queue
        self.source = source
        self.mute_source = mute_source
        self.thread = thread
