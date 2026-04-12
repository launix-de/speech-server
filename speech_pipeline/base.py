from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional
from uuid import uuid4

_LOGGER = logging.getLogger("stage")


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
        self.upstream: Optional[Stage] = None
        self.downstream: Optional[Stage] = None
        self.cancelled: bool = False
        self.closed: bool = False

    def set_upstream(self, up: Stage) -> Stage:
        self.upstream = up
        up.downstream = self
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
        if self.cancelled:
            return
        self.cancelled = True
        try:
            if self.upstream:
                self.upstream.cancel()
        except Exception:
            pass
        try:
            if self.downstream:
                self.downstream.cancel()
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

        Idempotent.
        """
        if self.closed:
            return
        self.closed = True
        try:
            self._on_close()
        except Exception as e:
            _LOGGER.warning("Stage %s _on_close raised: %s", type(self).__name__, e)
        # Propagate in BOTH directions — idempotent via ``closed`` flag.
        # A downstream-initiated close (e.g. HangupSink) flows upstream;
        # an upstream-initiated close (generator exhausted) flows down.
        for neighbour in (self.upstream, self.downstream):
            try:
                if neighbour is not None and not neighbour.closed:
                    neighbour.close()
            except Exception as e:
                _LOGGER.warning("Stage %s close propagation raised: %s",
                                type(self).__name__, e)

    def _on_close(self) -> None:
        """Subclass hook: release resources tied to this stage.

        Called exactly once by ``close()``.  Default: no-op.
        """

    def estimate_frames_24k(self) -> Optional[int]:
        return None

    def stream_pcm24k(self) -> Iterator[bytes]:
        if False:
            yield b""
