from __future__ import annotations

import math
import subprocess as _sp
import tempfile as _tempfile
import wave as _wave
from pathlib import Path
from typing import Iterator, Optional, Any, Callable

from .base import AudioFormat, Stage
from .util import estimate_f0_avg, ffmpeg_to_pcm16, read_wav_all_samples
from .FileFetcher import FileFetcher


class PitchAdjuster(Stage):
    def __init__(
        self,
        target_ref: Any,
        pitch_disable: bool,
        pitch_override_st: Optional[float],
        correction: float,
        bearer: str = "",
    ) -> None:
        super().__init__()
        self.target_ref = target_ref  # Path or readable
        self.pitch_disable = pitch_disable
        self.pitch_override = pitch_override_st
        self.correction = correction
        self._bearer = bearer
        self.applied_st: Optional[float] = None
        self._target_local: Optional[Path] = None
        self.f0_t: Optional[float] = None
        self.input_format = AudioFormat(24000, "s16le")
        self.output_format = AudioFormat(24000, "s16le")

    def _ensure_target_local(self) -> Optional[Path]:
        if isinstance(self.target_ref, Path):
            return self.target_ref
        if isinstance(self.target_ref, str):
            kind, value = FileFetcher._classify(self.target_ref)  # internal helper
            if kind == 'http':
                tmp = FileFetcher(value, bearer=self._bearer).to_local_tmp()
                if tmp:
                    self._target_local = tmp
                    return tmp
                return None
            return Path(value)
        if self._target_local is not None:
            return self._target_local
        try:
            import tempfile as _tempfile
            tmp = _tempfile.NamedTemporaryFile(prefix='pitch_target_', suffix='.wav', delete=False)
            p = Path(tmp.name); tmp.close()
            read = getattr(self.target_ref, 'read', None)
            if callable(read):
                with open(p, 'wb') as out:
                    while True:
                        buf = read(64 * 1024)
                        if not buf:
                            break
                        out.write(buf)
                self._target_local = p
                return p
        except Exception:
            return None
        return None

    def estimate_frames_24k(self) -> Optional[int]:
        return self.upstream.estimate_frames_24k() if self.upstream else None

    def stream_pcm24k(self) -> Iterator[bytes]:
        assert self.upstream is not None
        # Resolve target locally (string URL/file, Path, or readable) once
        target_local: Optional[Path] = None
        target_cleanup: Callable[[], None] = (lambda: None)
        try:
            if isinstance(self.target_ref, Path):
                target_local = self.target_ref
                target_cleanup = None
            elif isinstance(self.target_ref, str):
                ff = FileFetcher(self.target_ref, bearer=self._bearer)
                p, cleanup = ff.get_physical_file()
                target_local = p
                target_cleanup = cleanup
            else:
                tmp = _tempfile.NamedTemporaryFile(prefix='pitch_target_', suffix='.wav', delete=False)
                p = Path(tmp.name); tmp.close()
                read = getattr(self.target_ref, 'read', None)
                if callable(read):
                    with open(p, 'wb') as out:
                        while True:
                            buf = read(64 * 1024)
                            if not buf:
                                break
                            out.write(buf)
                    target_local = p
                    def _c():
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                    target_cleanup = _c
        except Exception:
            target_local = None
            target_cleanup = (lambda: None)
        # Precompute f0 target baseline if not already set
        if (self.f0_t is None) and target_local:
            try:
                sr_t, x_t = read_wav_all_samples(target_local)
                self.f0_t = estimate_f0_avg(sr_t, x_t)
            except Exception:
                self.f0_t = None
        try:
            for idx, pcm in enumerate(self.upstream.stream_pcm24k()):
                if self.cancelled:
                    break
                # decide pitch once on first chunk
                if (not self.pitch_disable) and (self.applied_st is None):
                    if (self.pitch_override is not None) and abs(self.pitch_override) > 0.05:
                        self.applied_st = float(self.pitch_override) * float(self.correction)
                    elif self.f0_t:
                        # compute f0 of this chunk
                        tmp = _tempfile.NamedTemporaryFile(prefix=f"pipe_pitch_src_{idx:04d}_", suffix=".wav", delete=False)
                        p = Path(tmp.name)
                        tmp.close()
                        with _wave.open(str(p), "wb") as ww:
                            ww.setnchannels(1)
                            ww.setsampwidth(2)
                            ww.setframerate(24000)
                            ww.writeframes(pcm)
                        try:
                            sr_v, x_v = read_wav_all_samples(p)
                            f0_v = estimate_f0_avg(sr_v, x_v)
                            if f0_v and f0_v > 0.0:
                                st_raw = 12.0 * math.log2(float(self.f0_t) / float(f0_v))
                                self.applied_st = float(st_raw) * float(self.correction)
                        except Exception:
                            pass
                        finally:
                            try:
                                p.unlink()
                            except Exception:
                                pass
                if (self.applied_st is not None) and abs(self.applied_st) > 0.1:
                    # write chunk to wav, run ffmpeg pitch, return pcm
                    tmpi = _tempfile.NamedTemporaryFile(prefix=f"pipe_pitch_in_{idx:04d}_", suffix=".wav", delete=False)
                    pi = Path(tmpi.name)
                    tmpi.close()
                    with _wave.open(str(pi), "wb") as ww:
                        ww.setnchannels(1)
                        ww.setsampwidth(2)
                        ww.setframerate(24000)
                        ww.writeframes(pcm)
                    tmpo = _tempfile.NamedTemporaryFile(prefix=f"pipe_pitch_out_{idx:04d}_", suffix=".wav", delete=False)
                    po = Path(tmpo.name)
                    tmpo.close()
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(pi),
                        "-filter:a",
                        f"rubberband=tempo=1.0:pitch={2.0 ** (self.applied_st / 12.0)}:formant=1",
                        str(po),
                    ]
                    try:
                        _sp.check_call(cmd)
                        with _wave.open(str(po), "rb") as wf:
                            yield wf.readframes(wf.getnframes())
                    except Exception:
                        # fallback: passthrough
                        yield pcm
                    finally:
                        for p in (pi, po):
                            try:
                                if p.exists():
                                    p.unlink()
                            except Exception:
                                pass
                else:
                    # default: passthrough
                    yield pcm
        finally:
            if target_cleanup is not None:
                target_cleanup()
