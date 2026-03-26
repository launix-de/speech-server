from __future__ import annotations

import subprocess as _sp
import time
from typing import Iterator, Optional

from .base import Stage
from .util import ffprobe_duration_sec


class AudioReader(Stage):
    def __init__(
        self,
        src_ref: str,
        bearer: str = "",
        chunk_seconds: float = 10.0,
        realtime: bool = False,
        prefill_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        from .base import AudioFormat
        self.src_ref = src_ref
        self.bearer = bearer
        self.chunk_seconds = float(chunk_seconds)
        self.realtime = bool(realtime)
        self.prefill_seconds = max(0.0, float(prefill_seconds))
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
            if self.realtime:
                # Telephony playback must track wall clock, otherwise ffmpeg
                # decodes file/HTTP input far faster than real-time and the
                # bounded mixer queue starts dropping chunks.
                chunk_seconds = max(0.02, min(self.chunk_seconds, 0.05))
            else:
                chunk_seconds = max(0.1, self.chunk_seconds)
            chunk_bytes = int(24000 * 2 * chunk_seconds)
            pending = bytearray()
            start_time = time.monotonic()
            if self.realtime and self.prefill_seconds > 0.0:
                start_time -= self.prefill_seconds
            emitted_seconds = 0.0
            while True:
                if self.cancelled:
                    break
                if proc.stdout is None:
                    break
                reader = getattr(proc.stdout, "read1", proc.stdout.read)
                buf = reader(chunk_bytes)
                if not buf:
                    break
                pending.extend(buf)
                emit_len = len(pending) & ~1
                if emit_len:
                    out = bytes(pending[:emit_len])
                    del pending[:emit_len]
                    if self.realtime:
                        emitted_seconds += len(out) / (24000 * 2)
                        wait = (start_time + emitted_seconds) - time.monotonic()
                        if wait > 0:
                            time.sleep(wait)
                    yield out
            if not self.cancelled:
                emit_len = len(pending) & ~1
                if emit_len:
                    out = bytes(pending[:emit_len])
                    if self.realtime:
                        emitted_seconds += len(out) / (24000 * 2)
                        wait = (start_time + emitted_seconds) - time.monotonic()
                        if wait > 0:
                            time.sleep(wait)
                    yield out
        finally:
            try:
                if proc and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
