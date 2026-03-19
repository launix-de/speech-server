from __future__ import annotations

import subprocess as _sp
from typing import Iterator, Optional

from .base import Stage
from .util import ffprobe_duration_sec


class AudioReader(Stage):
    def __init__(self, src_ref: str, bearer: str = "", chunk_seconds: float = 10.0) -> None:
        super().__init__()
        from .base import AudioFormat
        self.src_ref = src_ref
        self.bearer = bearer
        self.chunk_seconds = float(chunk_seconds)
        self.output_format = AudioFormat(24000, "s16le")

    def estimate_frames_24k(self) -> Optional[int]:
        d = ffprobe_duration_sec(self.src_ref)
        return int(d * 24000) if d and d > 0 else None

    def stream_pcm24k(self) -> Iterator[bytes]:
        cmd = ["ffmpeg", "-nostdin"]
        if (self.src_ref.startswith("http://") or self.src_ref.startswith("https://")) and self.bearer:
            cmd += ["-headers", f"Authorization: Bearer {self.bearer}\r\n"]
        cmd += ["-i", self.src_ref, "-f", "s16le", "-ac", "1", "-ar", "24000", "-loglevel", "error", "-"]
        proc = _sp.Popen(cmd, stdout=_sp.PIPE)
        try:
            chunk_bytes = int(24000 * 2 * max(0.1, self.chunk_seconds))
            while True:
                if self.cancelled:
                    break
                if proc.stdout is None:
                    break
                buf = proc.stdout.read(chunk_bytes)
                if not buf:
                    break
                yield buf
        finally:
            try:
                if proc and proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass

