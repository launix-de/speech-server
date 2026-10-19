from __future__ import annotations

from typing import Iterator, Optional

from .ResponseWriter import ResponseWriter
from .base import Stage


class RawResponseWriter(ResponseWriter):
    """HTTP sink for an existing audio container; preserves its bytes and MIME.

    Uses the same reference release and deadline lifecycle as the WAV writer.
    """

    def __init__(self, upstream=None, chunk_bytes: int = 64 * 1024,
                 timeout: Optional[float] = None) -> None:
        super().__init__(timeout=timeout)
        self.chunk_bytes = int(max(4096, chunk_bytes))
        if upstream is not None:
            if isinstance(upstream, Stage):
                upstream.pipe(self)
            else:
                self.set_upstream(upstream)

    def stream_pcm24k(self) -> Iterator[bytes]:
        if self.upstream is None or self.cancelled or self.closed:
            return
        source = self.upstream
        if isinstance(source, Stage):
            stream = source.stream_pcm24k()
        else:
            def read_chunks():
                while not self.cancelled:
                    chunk = source.read(self.chunk_bytes)
                    if not chunk:
                        break
                    yield chunk
            stream = read_chunks() if callable(getattr(source, "read", None)) else source.stream()
        if not self._track_stream(stream):
            return
        finished = False
        try:
            for chunk in stream:
                if self.cancelled:
                    break
                yield chunk
            finished = not self.cancelled
        finally:
            if not finished:
                self.cancel()
            try:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            finally:
                self.close()
