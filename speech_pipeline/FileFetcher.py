from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple
import os
from .base import AudioFormat, Stage


class FileFetcher(Stage):
    output_format = AudioFormat(0, "binary")

    def __init__(self, src_ref: str, bearer: str = "", chunk_bytes: int = 64 * 1024,
                 timeout: float = 30.0) -> None:
        super().__init__()
        self.src_ref = src_ref
        self.bearer = bearer
        self.chunk_bytes = int(max(4096, chunk_bytes))
        self._handle = None  # lazy-opened readable
        self.timeout = float(timeout)

    @staticmethod
    def _classify(src: str) -> Tuple[str, str]:
        if src.startswith("http://") or src.startswith("https://"):
            return ("http", src)
        return ("file", str(Path(src)))

    @staticmethod
    def build_ref(sound_id: str, template: str, base_dir: Path) -> str:
        """Return a URL or absolute file path from an id and a template.
        The template should contain "%s" where the id is inserted, or it will be appended.
        Local paths are resolved relative to base_dir.
        """
        try:
            target = template % sound_id
        except Exception:
            target = template.replace('%s', sound_id)
        if target.startswith('http://') or target.startswith('https://'):
            return target
        return str((base_dir / target).resolve())

    def get_physical_file(self) -> Tuple[Path, Callable[[], None]]:
        """Return a real filesystem Path for this ref and a cleanup function.
        - For http(s) refs, downloads to a temp file; cleanup removes it and closes streams.
        - For local files, returns the absolute Path; cleanup is a no-op.
        """
        kind, value = self._classify(self.src_ref)
        if kind == 'http':
            # Prefer in-memory file descriptor (memfd) on Linux; fallback to temp file
            # Ensure remote is opened
            h = self._open()
            # Try memfd
            try:
                fd = os.memfd_create("freevc_target", flags=0)  # type: ignore[attr-defined]
                try:
                    # Stream HTTP body into FD
                    while True:
                        buf = h.read(64 * 1024)
                        if not buf:
                            break
                        os.write(fd, buf)
                except Exception:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                    raise
                # Expose a stable path to the memfd
                mem_path = Path(f"/proc/self/fd/{fd}")
                def _cleanup_memfd():
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                    try:
                        self.close()
                    except Exception:
                        pass
                return mem_path, _cleanup_memfd
            except Exception:
                # Fallback: temp file on disk
                p = self.to_local_tmp()
                if not p:
                    raise FileNotFoundError(f"failed to fetch remote file: {self.src_ref}")
                def _cleanup_file():
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        self.close()
                    except Exception:
                        pass
                return Path(p), _cleanup_file
        # local file
        p = Path(value).resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p, (lambda: None)

    @staticmethod
    def fetch_to_temp(url: str, bearer: str = "") -> Optional[Path]:
        try:
            import tempfile, urllib.request as _urllib
            tmp = tempfile.NamedTemporaryFile(prefix='fetch_', suffix='.wav', delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            req = _urllib.Request(url)
            if bearer:
                req.add_header('Authorization', f'Bearer {bearer}')
            with _urllib.urlopen(req) as resp, open(tmp_path, 'wb') as out:
                while True:
                    data = resp.read(64 * 1024)
                    if not data:
                        break
                    out.write(data)
            return tmp_path
        except Exception:
            return None

    def _open(self):
        if self._handle is not None:
            return self._handle
        kind, value = self._classify(self.src_ref)
        if kind == 'http':
            import urllib.request as _urllib
            req = _urllib.Request(value)
            if self.bearer:
                req.add_header('Authorization', f'Bearer {self.bearer}')
            self._handle = _urllib.urlopen(req, timeout=self.timeout)
        else:
            self._handle = open(Path(value), 'rb')
        return self._handle

    # Python-readable stream interface
    def read(self, n: int = -1) -> bytes:
        return self._open().read(n)

    def stream(self) -> Iterator[bytes]:
        yield from self.stream_pcm24k()

    def stream_pcm24k(self) -> Iterator[bytes]:
        h = self._open()
        try:
            while not self.cancelled and not self.closed:
                buf = h.read(self.chunk_bytes)
                if not buf:
                    break
                yield buf
        finally:
            self.close()

    def _on_close(self) -> None:
        try:
            if self._handle is not None:
                self._handle.close()
        except Exception:
            pass

    def to_local_tmp(self) -> Optional[Path]:
        try:
            import tempfile as _tempfile
            tmp = _tempfile.NamedTemporaryFile(prefix='fetch_local_', suffix='.wav', delete=False)
            p = Path(tmp.name); tmp.close()
            read = getattr(self, 'read', None)
            if callable(read):
                with open(p, 'wb') as out:
                    while True:
                        buf = read(64 * 1024)
                        if not buf:
                            break
                        out.write(buf)
                return p
        except Exception:
            return None
        return None

    # resolve_to_local removed; constructor provides readable stream for both http and file
