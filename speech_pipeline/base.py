from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional
from uuid import uuid4

_LOGGER = logging.getLogger("stage")
_GRAPH_LOCK = threading.RLock()


@dataclass(frozen=True)
class AudioFormat:
    """Describes the audio format at a stage boundary.

    sample_rate: Hz (0 for non-audio like text/ndjson)
    encoding:    's16le', 'u8', 'text', 'ndjson'
    channels:    number of channels (default 1 = mono)
    """
    sample_rate: int
    encoding: str
    channels: int = 1


def _build_converter_chain(src: AudioFormat, dst: AudioFormat) -> List[Stage]:
    """Build a chain of converter stages to bridge src -> dst format.

    Only handles audio encoding and sample rate conversions.
    Non-audio formats (text, ndjson) are not auto-converted.
    """
    if src == dst:
        return []

    audio_encodings = {"s16le", "u8"}
    if src.encoding not in audio_encodings or dst.encoding not in audio_encodings:
        return []

    # Lazy imports to avoid circular dependencies
    from .EncodingConverter import EncodingConverter
    from .SampleRateConverter import SampleRateConverter

    chain: List[Stage] = []
    cur_encoding = src.encoding
    cur_rate = src.sample_rate

    need_encode = cur_encoding != dst.encoding
    need_resample = cur_rate != dst.sample_rate and cur_rate > 0 and dst.sample_rate > 0

    if need_encode and need_resample:
        # SampleRateConverter works on s16le, so ensure s16le before resampling
        if cur_encoding != "s16le":
            chain.append(EncodingConverter(cur_encoding, "s16le"))
            cur_encoding = "s16le"
        chain.append(SampleRateConverter(cur_rate, dst.sample_rate))
        cur_rate = dst.sample_rate
        if dst.encoding != "s16le":
            chain.append(EncodingConverter("s16le", dst.encoding))
    elif need_encode:
        chain.append(EncodingConverter(cur_encoding, dst.encoding))
    elif need_resample:
        chain.append(SampleRateConverter(cur_rate, dst.sample_rate))

    _LOGGER.debug(
        "Auto-inserted %d converter(s): %s -> %s",
        len(chain), src, dst,
    )
    return chain


class Stage:
    input_format: Optional[AudioFormat] = None
    output_format: Optional[AudioFormat] = None

    def __init__(self) -> None:
        self.id: str = uuid4().hex[:8]
        self._upstream: Optional[Stage] = None
        self._consumers: dict[Stage, None] = {}
        self._references = 0
        self._close_callbacks = []
        self._close_complete = False
        self.cancelled: bool = False
        self.closed: bool = False

    @property
    def upstream(self) -> Optional[Stage]:
        return self._upstream

    @upstream.setter
    def upstream(self, up: Optional[Stage]) -> None:
        self.set_upstream(up)

    @property
    def downstreams(self) -> tuple[Stage, ...]:
        """Connected consumers, each holding one reference to this stage."""
        with _GRAPH_LOCK:
            return tuple(self._consumers)

    @property
    def downstream(self) -> Optional[Stage]:
        """Compatibility view for existing runners of linear pipelines."""
        with _GRAPH_LOCK:
            return next(reversed(self._consumers), None)

    @downstream.setter
    def downstream(self, down: Optional[Stage]) -> None:
        if down is not None:
            down.set_upstream(self)
        else:
            current = self.downstream
            if current is not None:
                current.set_upstream(None)

    @property
    def reference_count(self) -> int:
        with _GRAPH_LOCK:
            return len(self._consumers) + self._references

    def retain(self) -> StageReference:
        """Own a shared stage across queue/mixer boundaries or outside a pipe.

        Release the returned handle when that owner disappears. Data fan-out
        still uses AudioTee or mixer queues; retaining does not duplicate audio.
        """
        with _GRAPH_LOCK:
            if self.closed or self.cancelled:
                raise RuntimeError("Cannot retain a finished stage")
            self._references += 1
        return StageReference(self)

    def add_close_callback(self, callback) -> None:
        """Release an external resource once this stage has closed."""
        with _GRAPH_LOCK:
            if not self._close_complete:
                self._close_callbacks.append(callback)
                return
        callback()

    def set_upstream(self, up: Optional[Stage]) -> Stage:
        with _GRAPH_LOCK:
            old = self._upstream
            if old is up:
                return self
            if up is not None and (self.closed or self.cancelled):
                raise RuntimeError("Cannot connect a finished stage")
            if issubclass(type(up), Stage):
                if up.closed or up.cancelled:
                    raise RuntimeError("Cannot consume a finished stage")
                up._consumers[self] = None
            self._upstream = up
            if issubclass(type(old), Stage):
                old._consumers.pop(self, None)
        # Establish the new edge before releasing the old one. This makes
        # removing/replacing a processor safe even with a shared upstream.
        if issubclass(type(old), Stage):
            old._close_if_unused()
        return self

    def pipe(self, next_stage: Stage) -> Stage:
        """Connect this stage to next_stage.

        If both stages declare audio formats and they don't match,
        converter stages (EncodingConverter, SampleRateConverter) are
        automatically inserted between them.
        """
        src_fmt = self.output_format
        dst_fmt = next_stage.input_format

        if src_fmt is not None and dst_fmt is not None and src_fmt != dst_fmt:
            converters = _build_converter_chain(src_fmt, dst_fmt)
            if converters:
                current = self
                for conv in converters:
                    current = conv.set_upstream(current)
                return next_stage.set_upstream(current)

        return next_stage.set_upstream(self)

    def cancel(self) -> None:
        with _GRAPH_LOCK:
            if self.cancelled:
                return
            self.cancelled = True
            up = self._upstream
            consumers = tuple(self._consumers)
            cancel_up = (up is not None and not issubclass(type(up), Stage)) or (
                issubclass(type(up), Stage) and up._references == 0 and not any(
                    child is not self and not child.cancelled and not child.closed
                    for child in up._consumers
                )
            )
        try:
            if cancel_up:
                up.cancel()
        except Exception:
            pass
        for child in consumers:
            try:
                child.cancel()
            except Exception:
                pass

    def close(self) -> None:
        """Orderly teardown — signal that upstream has completed.

        Distinct from ``cancel()``:

        * ``cancel()`` = abort immediately, skip remaining buffered data
          (forced teardown; propagates up AND down).
        * ``close()`` = orderly finish; in-flight data must drain first.
          The caller invokes ``close()`` AFTER upstream's generator has
          exhausted.  Default implementation propagates the signal
          downstream (letting the drain cascade continue) and invokes
          the subclass hook ``_on_close()`` for resource release.

        Closing a consumer releases its upstream edge. Only an upstream with
        no remaining consumers or retained owners is closed in turn. Explicitly
        closing a producer still closes all its outputs. Idempotent.
        """
        self._close(only_if_unused=False)

    def _close_if_unused(self) -> None:
        self._close(only_if_unused=True)

    def _close(self, *, only_if_unused: bool) -> None:
        with _GRAPH_LOCK:
            if self.closed or (only_if_unused and self.reference_count):
                return
            self.closed = True
            consumers = tuple(self._consumers)
        try:
            self._on_close()
        except Exception as e:
            _LOGGER.warning("Stage %s _on_close raised: %s", type(self).__name__, e)
        # Hooks run outside the graph lock: mixer hooks may acquire their own
        # locks or join workers that also need to release graph references.
        with _GRAPH_LOCK:
            up = self._upstream
            self._upstream = None
            if issubclass(type(up), Stage):
                up._consumers.pop(self, None)
        try:
            if issubclass(type(up), Stage):
                up._close_if_unused()
            elif up is not None:
                up.close()
        except Exception as e:
            _LOGGER.warning("Stage %s upstream close raised: %s", type(self).__name__, e)
        for neighbour in consumers:
            try:
                if neighbour is not None and not neighbour.closed:
                    neighbour.close()
            except Exception as e:
                _LOGGER.warning("Stage %s close propagation raised: %s",
                                type(self).__name__, e)

        with _GRAPH_LOCK:
            self._close_complete = True
            callbacks, self._close_callbacks = self._close_callbacks, []
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                _LOGGER.warning("Stage %s close callback raised: %s", type(self).__name__, e)

    def _on_close(self) -> None:
        """Subclass hook: release resources tied to this stage.

        Called exactly once by ``close()``.  Default: no-op.
        """

    def estimate_frames_24k(self) -> Optional[int]:
        return None

    def stream_pcm24k(self) -> Iterator[bytes]:
        if False:
            yield b""


class StageReference:
    """An explicit, idempotently releasable owner of a shared stage."""

    def __init__(self, stage: Stage) -> None:
        self._stage: Optional[Stage] = stage

    def release(self) -> None:
        with _GRAPH_LOCK:
            stage = self._stage
            if stage is None:
                return
            self._stage = None
            stage._references -= 1
        stage._close_if_unused()
