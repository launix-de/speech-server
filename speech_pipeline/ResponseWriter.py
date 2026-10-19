from __future__ import annotations

import logging
import math
import threading
import tempfile
from typing import Any, Iterator, Optional

from .base import Stage

_LOGGER = logging.getLogger("piper-multi-server")


class ResponseWriter(Stage):
    """WAV HTTP sink with an optional deadline, including unconsumed responses.

    ``stream()`` and the upstream constructor argument remain compatible with
    older callers. New pipelines can use ``source.pipe(ResponseWriter(...))``.
    Timeout/disconnect cancels this branch, then releases its graph references.
    Shared upstream nodes remain alive while other consumers/owners retain them.
    Native inference cancellation is cooperative; it cannot interrupt a C call.
    """

    def __init__(self, upstream: Optional[Stage] = None,
                 est_frames_24k: Optional[int] = None,
                 max_chunk_bytes: Optional[int] = None,
                 timeout: Optional[float] = None, buffered: bool = False,
                 estimated_length: bool = True) -> None:
        super().__init__()
        self.est_frames = est_frames_24k
        self.max_chunk_bytes = None
        if max_chunk_bytes is not None:
            self.max_chunk_bytes = int(max_chunk_bytes) & ~1
            if self.max_chunk_bytes < 2:
                raise ValueError("max_chunk_bytes must hold at least one s16 sample")
        self.sample_rate = 24000
        self.buffered = buffered
        self.estimated_length = estimated_length
        self.timeout = None if timeout is None else float(timeout)
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be a positive finite number")
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._stream = None
        self._buffer = None
        self._stream_active = False
        if upstream is not None:
            upstream.pipe(self)

    def set_upstream(self, up: Optional[Stage]) -> Stage:
        super().set_upstream(up)
        if up is not None:
            fmt = getattr(up, "output_format", None)
            self.sample_rate = fmt.sample_rate if fmt and fmt.sample_rate > 0 else 24000
            with self._timer_lock:
                if self.timeout is not None and self._timer is None and not self.closed:
                    self._timer = threading.Timer(self.timeout, self.cancel)
                    self._timer.daemon = True
                    self._timer.start()
        return self

    def _on_close(self) -> None:
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._close_stream()
        if self._buffer is not None:
            self._buffer.close()

    def _track_stream(self, stream) -> bool:
        with self._stream_lock:
            self._stream = stream
        if self.closed or self.cancelled:
            self._close_stream()
            return False
        return True

    def _close_stream(self) -> None:
        with self._stream_lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            except ValueError:
                # A generator currently running in the request thread cannot
                # be closed by the timer thread. The cancelled flag makes
                # that thread unwind it when its current operation returns.
                _LOGGER.debug("writer: deferring close of an executing generator")
            except Exception:
                _LOGGER.warning("writer: upstream iterator cleanup failed", exc_info=True)

    def cancel(self) -> None:
        super().cancel()
        self.close()

    def close(self) -> None:
        # Upstream EOF may close its consumers while this sink still has
        # buffered bytes to deliver. The deadline stays armed until delivery.
        if self._stream_active and not self.cancelled:
            return
        super().close()

    def estimate_frames_24k(self) -> Optional[int]:
        return self.est_frames if self.est_frames is not None else (
            self.upstream.estimate_frames_24k() if self.upstream else None
        )

    def _data_size(self) -> int:
        frames = self.estimate_frames_24k()
        if frames is None or frames <= 0:
            frames = 30 * self.sample_rate
        size = max(0, int(int(frames) * 2 * 1.05))
        return min(size + size % 2, 0xFFFFFFFE)

    def stream(self) -> Iterator[bytes]:
        yield from self.stream_pcm24k()

    def stream_pcm24k(self) -> Iterator[bytes]:
        source = self.upstream
        if source is None or self.cancelled or self.closed:
            return
        self._stream_active = True
        stream = source.stream_pcm24k()
        if not self._track_stream(stream):
            self._stream_active = False
            return
        finished = False
        try:
            sr = self.sample_rate
            if self.buffered:
                # Keep native WAV framing exact without keeping long clips in
                # RAM or appending guessed silence to short dialog lines.
                self._buffer = tempfile.SpooledTemporaryFile(max_size=2 * 1024 * 1024)
                for pcm in stream:
                    if self.cancelled:
                        return
                    self._buffer.write(pcm)
                if self.cancelled:
                    return
                data_size = self._buffer.tell()
                self._buffer.seek(0)
            else:
                data_size = self._data_size() if self.estimated_length else 0xFFFFFFDA
            header = (
                b"RIFF" + min(36 + data_size, 0xFFFFFFFF).to_bytes(4, "little")
                + b"WAVEfmt " + (16).to_bytes(4, "little")
                + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
                + sr.to_bytes(4, "little") + (sr * 2).to_bytes(4, "little")
                + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
                + b"data" + data_size.to_bytes(4, "little")
            )
            yield header
            total = 0
            output = iter(lambda: self._buffer.read(64 * 1024), b"") if self.buffered else stream
            for pcm in output:
                if self.cancelled:
                    break
                total += len(pcm)
                size = self.max_chunk_bytes or max(2, len(pcm))
                for offset in range(0, len(pcm), size):
                    if self.cancelled:
                        break
                    yield pcm[offset:offset + size]
            # Preserve the legacy finite WAV header without allocating a large
            # silence buffer. HTTP cleanup must also work while padding yields.
            while self.estimated_length and not self.buffered and not self.cancelled and total < data_size:
                count = min(data_size - total, 64 * 1024)
                yield bytes(count)
                total += count
            if total > data_size:
                _LOGGER.warning("writer: streamed bytes=%d exceed declared size=%d", total, data_size)
            finished = not self.cancelled
        finally:
            if not finished:
                self.cancel()
            try:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            finally:
                self._stream_active = False
                if self._buffer is not None:
                    self._buffer.close()
                self.close()

    def apply_headers(self, resp: Any) -> None:
        resp.headers["Content-Type"] = "audio/wav"
        resp.headers["Content-Disposition"] = "inline"
        if self.estimated_length and not self.buffered:
            resp.headers["Content-Length"] = str(44 + self._data_size())
