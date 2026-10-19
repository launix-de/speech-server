from __future__ import annotations

import os
from typing import Optional
import threading
import logging

_CoquiTTS = None
_backend_import_attempted = False
_backend_import_lock = threading.Lock()
_LOGGER = logging.getLogger("freevc")


def _load_backend():
    """Import optional PyTorch/Coqui only when conversion is requested."""
    global _CoquiTTS, _backend_import_attempted
    with _backend_import_lock:
        if not _backend_import_attempted:
            _backend_import_attempted = True
            try:
                os.environ.setdefault("USE_TF", "0")
                from TTS.api import TTS
                _CoquiTTS = TTS
            except ImportError:
                _LOGGER.warning("FreeVC unavailable: install speech-pipeline[vc] for reference voices")
        return _CoquiTTS


_singleton_model = None
_singleton_init_lock = threading.Lock()
conversion_lock = threading.Lock()


class FreeVCService:
    def __init__(self, model_name: str = "voice_conversion_models/multilingual/vctk/freevc24", device_pref_env: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device_pref_env = device_pref_env or os.environ.get("FREEVC_DEVICE") or os.environ.get("TTS_DEVICE") or ""
        self._model = None

    def _init_to(self, device: str):
        backend = _load_backend()
        if backend is None:
            raise RuntimeError("VC unavailable: TTS not installed")
        if device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        t = backend(self.model_name, gpu=False)
        try:
            t.to(device)
        except Exception:
            if device != "cpu":
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
                t.to("cpu")
        return t

    def get_model(self):
        if self._model is not None:
            return self._model
        want_cuda = (self.device_pref_env or "").lower() == "cuda"
        if want_cuda:
            try:
                self._model = self._init_to("cuda")
                return self._model
            except Exception:
                pass
        self._model = self._init_to("cpu")
        return self._model

    def convert_to_file(self, source_wav: str, target_wav: str, file_path: str) -> None:
        self.get_model().voice_conversion_to_file(source_wav=source_wav, target_wav=target_wav, file_path=file_path)  # type: ignore


def get_freevc_model() -> Optional[object]:
    """Return a process-wide singleton FreeVC model, preferring CPU unless FREEVC_DEVICE=cuda.
    Returns None if TTS is not installed or init fails.
    """
    global _singleton_model
    if _singleton_model is not None:
        return _singleton_model
    with _singleton_init_lock:
        if _singleton_model is not None:
            return _singleton_model
        try:
            svc = FreeVCService()
            m = svc.get_model()
            _singleton_model = m
            return m
        except Exception:
            _LOGGER.exception("FreeVC model initialization failed")
            return None
