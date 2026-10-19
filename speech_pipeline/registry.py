from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from piper import PiperVoice, SynthesisConfig
except Exception as e:  # pragma: no cover
    raise RuntimeError("Piper must be importable before using TTSRegistry") from e


@dataclass
class VoiceInfo:
    model_id: str
    path: Path
    espeak_voice: str
    sample_rate: int
    num_speakers: int
    speaker_id_map: Dict[str, int]


def discover_voices(scan_dirs: Iterable[Path]) -> Dict[str, Path]:
    voices: Dict[str, Path] = {}
    for d in scan_dirs:
        if not d.exists():
            continue
        for onnx in d.rglob("*.onnx"):
            model_id = onnx.name[:-5] if onnx.name.endswith(".onnx") else onnx.stem
            voices.setdefault(model_id, onnx)
    return voices


def load_voice_info(model_path: Path) -> VoiceInfo:
    # Reading voice metadata must not construct another ONNX session and its
    # thread pool. It is also used by /voices and language discovery.
    from piper.config import PiperConfig
    with Path(str(model_path) + ".json").open() as config_file:
        cfg = PiperConfig.from_dict(json.load(config_file))
    return VoiceInfo(
        model_id=model_path.stem,
        path=model_path,
        espeak_voice=cfg.espeak_voice,
        sample_rate=cfg.sample_rate,
        num_speakers=cfg.num_speakers,
        speaker_id_map=cfg.speaker_id_map,
    )


class TTSRegistry:
    def __init__(self, voices_path: Path | str, use_cuda: bool = False, voice_ttl_seconds: int = 7200, voice_cache_max: int = 64) -> None:
        self.voices_dir = Path(voices_path).resolve()
        self.use_cuda = bool(use_cuda)
        self.voice_ttl = int(max(0, voice_ttl_seconds))
        self.cache_max = int(max(1, voice_cache_max))
        self.index: Dict[str, Path] = discover_voices([self.voices_dir])
        self.loaded: Dict[str, PiperVoice] = {}
        self.infos: Dict[str, VoiceInfo] = {}
        self.last_used: Dict[str, float] = {}

    def refresh_index(self) -> None:
        self.index.update(discover_voices([self.voices_dir]))

    def _mark_used(self, model_id: str) -> None:
        self.last_used[model_id] = time.time()

    def _evict(self) -> None:
        # TTL eviction
        now = time.time()
        for mid, last in list(self.last_used.items()):
            if (now - float(last)) > self.voice_ttl:
                self.loaded.pop(mid, None)
                self.infos.pop(mid, None)
                self.last_used.pop(mid, None)
        # LRU size cap
        if len(self.loaded) > self.cache_max:
            order = sorted(self.last_used.items(), key=lambda kv: kv[1])
            overflow = len(self.loaded) - self.cache_max
            for mid, _ in order:
                if overflow <= 0:
                    break
                if mid in self.loaded:
                    self.loaded.pop(mid, None)
                    self.infos.pop(mid, None)
                    self.last_used.pop(mid, None)
                    overflow -= 1

    def ensure_loaded(self, model_id: str) -> PiperVoice:
        self._evict()
        v = self.loaded.get(model_id)
        if v is not None:
            self._mark_used(model_id)
            return v
        path = self.index.get(model_id)
        if not path:
            self.refresh_index()
            path = self.index.get(model_id)
        if not path:
            raise KeyError(f"Voice not found: {model_id}")
        voice = PiperVoice.load(path, use_cuda=self.use_cuda)
        self.loaded[model_id] = voice
        self._mark_used(model_id)
        try:
            self.infos[model_id] = load_voice_info(path)
        except Exception:
            pass
        return voice

    def best_for_lang(self, lang: str) -> Optional[str]:
        if not lang:
            return None
        import re
        m = re.match(r"([a-zA-Z]{2,3})", lang)
        lang2 = m.group(1).lower() if m else lang.lower()
        for mid, info in self.infos.items():
            if info.espeak_voice.lower().startswith(lang2):
                return mid
        for mid, onnx in self.index.items():
            try:
                info = load_voice_info(onnx)
            except Exception:
                continue
            self.infos[mid] = info
            if info.espeak_voice.lower().startswith(lang2):
                return mid
        return None

    def create_synthesis_config(self, voice: PiperVoice, params: Dict[str, Any]) -> SynthesisConfig:
        def _as_float(v: Any, fallback: float) -> float:
            try:
                if v is None:
                    return float(fallback)
                return float(v)
            except Exception:
                return float(fallback)
        # speaker selection
        speaker_id: Optional[int] = None
        sid_val = params.get("speaker_id")
        if sid_val is not None and str(sid_val).strip() != "":
            try:
                speaker_id = int(sid_val)
            except Exception:
                speaker_id = None
        if (voice.config.num_speakers > 1) and (speaker_id is None):
            speaker = params.get("speaker")
            if speaker:
                try:
                    speaker_id = voice.config.speaker_id_map.get(str(speaker))
                except Exception:
                    speaker_id = None
            if speaker_id is None:
                speaker_id = 0
        if (speaker_id is not None) and (speaker_id >= voice.config.num_speakers):
            speaker_id = 0

        # Fallback defaults if voice.config fields are None
        ls_def = getattr(voice.config, 'length_scale', None) or 1.0
        ns_def = getattr(voice.config, 'noise_scale', None) or 0.667
        nws_def = getattr(voice.config, 'noise_w_scale', None) or 0.8
        return SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=_as_float(params.get("length_scale"), ls_def),
            noise_scale=_as_float(params.get("noise_scale"), ns_def),
            noise_w_scale=_as_float(params.get("noise_w_scale"), nws_def),
        )

    def create_tts_stream(self, model_id: str, text: str, params: Dict[str, Any]):
        voice = self.ensure_loaded(model_id)
        syn = self.create_synthesis_config(voice, params)
        # lazy import to avoid hard dependency at module import
        from .TTSProducer import TTSProducer
        sentence_silence = float(params.get("sentence_silence", 0.0) or 0.0)
        chunk_seconds = float(params.get("chunk_seconds", 10.0) or 10.0)
        return TTSProducer(voice=voice, syn_config=syn, text=text, sentence_silence=sentence_silence, chunk_seconds=chunk_seconds)
