from __future__ import annotations

import os
import tempfile as _tempfile
import wave as _wave
from pathlib import Path
from typing import Callable, Iterator, Optional, Any

from .base import AudioFormat, Stage
from .util import ffmpeg_to_pcm16
from .vc_service import get_freevc_model, conversion_lock
from .FileFetcher import FileFetcher


class VCConverter(Stage):
    def __init__(self, target_ref: Any, vc_convert: Optional[Callable[[str, str, str], None]] = None, bearer: str = "") -> None:
        super().__init__()
        self.target_ref = target_ref  # Path or readable stream
        self.vc_convert = vc_convert
        self._vc_model = None
        self._target_local: Optional[Path] = None
        self._bearer = bearer
        self.input_format = AudioFormat(24000, "s16le")
        self.output_format = AudioFormat(24000, "s16le")

    def _ensure_internal_model(self):
        if self._vc_model is not None:
            return self._vc_model
        try:
            from TTS.api import TTS as _CoquiTTS  # type: ignore
        except Exception:
            return None
        # Device preference via env, default to CPU to avoid accidental CUDA usage
        dev_pref = (os.environ.get("FREEVC_DEVICE") or os.environ.get("TTS_DEVICE") or "").lower()
        want_cuda = (dev_pref == "cuda")
        if not want_cuda:
            # Force CPU visibility before model construction so internal modules pick CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            try:
                t = _CoquiTTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=False)  # type: ignore
                try:
                    t.to("cpu")
                except Exception:
                    pass
                self._vc_model = t
                return t
            except Exception:
                return None
        # Try CUDA, then hard fall back to a fresh CPU instance if anything fails
        try:
            t = _CoquiTTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=False)  # type: ignore
            t.to("cuda")
            self._vc_model = t
            return t
        except Exception:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                t = _CoquiTTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=False)  # type: ignore
                try:
                    t.to("cpu")
                except Exception:
                    pass
                self._vc_model = t
                return t
            except Exception:
                return None

    def _ensure_target_local(self) -> Optional[Path]:
        if isinstance(self.target_ref, Path):
            return self.target_ref
        if isinstance(self.target_ref, str):
            # Accept direct URL or file path string
            kind, value = FileFetcher._classify(self.target_ref)
            if kind == 'http':
                tmp = FileFetcher(value, bearer=self._bearer).to_local_tmp()
                if tmp:
                    self._target_local = tmp
                    return tmp
                return None
            return Path(value)
        if self._target_local is not None:
            return self._target_local
        # write readable stream to temp file
        try:
            tmp = _tempfile.NamedTemporaryFile(prefix='vc_target_', suffix='.wav', delete=False)
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
        # Resolve target locally once and initialize model once
        target_local: Optional[Path] = None
        target_cleanup: Callable[[], None] = (lambda: None)
        try:
            if isinstance(self.target_ref, Path):
                target_local = self.target_ref
                target_cleanup = None
            elif isinstance(self.target_ref, str):
                # Use FileFetcher to resolve HTTP/file refs
                ff = FileFetcher(self.target_ref, bearer=self._bearer)
                p, cleanup = ff.get_physical_file()
                target_local = p
                target_cleanup = cleanup
            else:
                # Readable stream: spill to temp file
                tmp = _tempfile.NamedTemporaryFile(prefix='vc_target_', suffix='.wav', delete=False)
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
        model = self._vc_model if (self.vc_convert is None) else None
        if (self.vc_convert is None) and (model is None):
            model = get_freevc_model()
        for idx, pcm in enumerate(self.upstream.stream_pcm24k()):
            if self.cancelled:
                break
            # write PCM to WAV @24k, run VC (or passthrough if unavailable)
            tmp_w = _tempfile.NamedTemporaryFile(prefix=f"pipe_vc_src_{idx:04d}_", suffix=".wav", delete=False)
            w_path = Path(tmp_w.name)
            tmp_w.close()
            with _wave.open(str(w_path), "wb") as ww:
                ww.setnchannels(1)
                ww.setsampwidth(2)
                ww.setframerate(24000)
                ww.writeframes(pcm)
            tmp_v = _tempfile.NamedTemporaryFile(prefix=f"pipe_vc_out_{idx:04d}_", suffix=".wav", delete=False)
            v_path = Path(tmp_v.name)
            tmp_v.close()
            try:
                if self.vc_convert is not None:
                    if target_local is None:
                        raise RuntimeError("VC target unavailable")
                    self.vc_convert(str(w_path), str(target_local), str(v_path))
                else:
                    if model is None:
                        raise RuntimeError("VC not available; passthrough")
                    if target_local is None:
                        raise RuntimeError("VC target unavailable")
                    # serialize FreeVC calls to avoid thread-safety issues
                    try:
                        with conversion_lock:
                            model.voice_conversion_to_file(source_wav=str(w_path), target_wav=str(target_local), file_path=str(v_path))  # type: ignore
                    except Exception:
                        raise
            except Exception:
                v_path = w_path
            # Normalize to PCM16@24k
            tmp_p = _tempfile.NamedTemporaryFile(prefix=f"pipe_vc_pcm_{idx:04d}_", suffix=".wav", delete=False)
            p_path = Path(tmp_p.name)
            tmp_p.close()
            if not ffmpeg_to_pcm16(v_path, p_path, sample_rate=24000):
                p_path = v_path
            try:
                with _wave.open(str(p_path), "rb") as wf:
                    yield wf.readframes(wf.getnframes())
            finally:
                for p in (w_path, v_path, p_path):
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
        # cleanup target
        try:
            target_cleanup()
        except Exception:
            pass
