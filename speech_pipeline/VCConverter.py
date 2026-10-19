from __future__ import annotations

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
        try:
            model = self._vc_model if self.vc_convert is None else None
            if self.vc_convert is None and model is None:
                model = get_freevc_model()
            if self.vc_convert is None and model is None:
                raise RuntimeError("Voice conversion backend unavailable")
            if target_local is None:
                raise RuntimeError("Voice conversion target unavailable")
            for pcm in self.upstream.stream_pcm24k():
                if self.cancelled:
                    break
                with _tempfile.TemporaryDirectory(prefix="pipe_vc_") as directory:
                    source = Path(directory) / "source.wav"
                    converted = Path(directory) / "converted.wav"
                    normalized = Path(directory) / "normalized.wav"
                    with _wave.open(str(source), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(24000)
                        wav.writeframes(pcm)
                    if self.vc_convert is not None:
                        self.vc_convert(str(source), str(target_local), str(converted))
                    else:
                        with conversion_lock:
                            if self.cancelled:
                                break
                            model.voice_conversion_to_file(source_wav=str(source),
                                target_wav=str(target_local), file_path=str(converted))
                    if self.cancelled:
                        break
                    if not ffmpeg_to_pcm16(converted, normalized, sample_rate=24000):
                        raise RuntimeError("Cannot normalize converted voice audio")
                    with _wave.open(str(normalized), "rb") as wav:
                        yield wav.readframes(wav.getnframes())
        finally:
            if target_cleanup is not None:
                target_cleanup()
