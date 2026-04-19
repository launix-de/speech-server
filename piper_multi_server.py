#!/usr/bin/env python3
"""
Multi-voice Piper HTTP server with realtime streaming and CORS.

Install (Python packages)
- Flask (web server)
    pip install Flask
- Piper (Python bindings for ONNX voices)
    Option A: install your local sources
      cd /home/carli/sources/piper
      pip install -e .
    Option B: if publishing is available, install from PyPI (name may differ)

Runtime dependencies (typical)
- onnxruntime or onnxruntime-gpu (choose one; GPU recommended)
    pip install onnxruntime
    # or
    pip install onnxruntime-gpu
- Coqui TTS (for optional FreeVC voice conversion)
    pip install TTS
- PyTorch (required by Coqui TTS; choose the right build for your CUDA/CPU)
    # CPU example
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    # CUDA example (adjust version)
    pip install torch --index-url https://download.pytorch.org/whl/cu121
- espeak-ng (system package) for phonemization used by many voices
    Debian/Ubuntu: sudo apt-get install espeak-ng

Quick start
- Install Python deps:
    pip install -r tts-piper/requirements.txt
    pip install -e /home/carli/sources/piper

Features
- Serves multiple voices/languages discovered from one or more scan directories.
- Similar endpoints to Piper's built-in http_server, plus CORS and language selection.
- Always streams realtime audio using a pipeline of stages.

Endpoints
- GET  /healthz                 -> 200 OK
- GET  /voices                  -> JSON map of available voices and metadata
- POST /                        -> JSON {text, voice?, lang?, speaker?, speaker_id?, length_scale?, noise_scale?, noise_w_scale?, sentence_silence?, voice2?, sound?, pitch_st?, pitch_factor?, pitch_disable?}

Run examples
  # auto-scan common folders for *.onnx voices
  python3 tts-piper/piper_multi_server.py --host 0.0.0.0 --port 5000

  # explicit scan dir
  python3 tts-piper/piper_multi_server.py --scan-dir . --scan-dir ../voices --scan-dir ../voices-piper

Maintainer notes
- Keep the request/response shape aligned with /home/carli/sources/piper/src/piper/http_server.py
- If Piper adds fields to SynthesisConfig or the API, mirror them here.
- CORS is enabled globally (after_request) to simplify browser usage.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import mimetypes, os

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_sock import Sock

# Ensure Piper sources are discoverable if installed from local sources
import sys as _sys
from pathlib import Path as _Path
_candidates = [
    _Path('/home/carli/sources/piper/src'),
    (_Path(__file__).resolve().parents[2] / 'sources' / 'piper' / 'src'),
]
for _p in _candidates:
    try:
        if _p.exists():
            _sys.path.insert(0, str(_p))
    except Exception:
        pass

# Optional: FreeVC via Coqui TTS for voice conversion
# FreeVC availability and initialization handled inside VCConverter stage


_LOGGER = logging.getLogger("piper-multi-server")

# Baseline constant for Thorsten (low, slow)
# Calibrated via calibrate_baseline.py on your x.wav
BASE_F0_HZ = 134.45

# Post-VC pitch damping factor (0..1). VC already nudges pitch; apply only a portion.
PITCH_CORRECTION = 0.5

# Default: do not disable pitch; can be overridden per-request via disable_pitch=1
PITCH_DISABLE_DEFAULT = False

# Chunk size used for VC/ffmpeg processing steps (seconds)
CHUNKSIZE_SECONDS = 10.0

def create_app(args: argparse.Namespace) -> Flask:
    # Voices live in a single folder (default: ./voices-piper). Allow override via --voices-path.
    # Back-compat: --scan-dir (single) behaves like --voices-path
    voices_arg = getattr(args, 'scan_dir', None) or getattr(args, 'voices_path', 'voices-piper')
    voices_dir = Path(voices_arg).resolve()
    _LOGGER.info("Voices dir: %s", voices_dir)
    # Use TTSRegistry for voice discovery, caching, and loading
    from lib.registry import TTSRegistry, load_voice_info, VoiceInfo  # type: ignore
    registry = TTSRegistry(voices_dir, use_cuda=args.cuda,
                           voice_ttl_seconds=int(getattr(args, 'voice_ttl_seconds', 7200)),
                           voice_cache_max=int(getattr(args, 'voice_cache_max', 64)))
    _LOGGER.info("Discovered %d voices", len(registry.index))
    # VC handled inside VCConverter; no global service here

    def ensure_loaded(model_id: str):
        return registry.ensure_loaded(model_id)

    # Preload default voice if provided
    default_model_id: Optional[str] = None
    if args.model:
        default_model_path = Path(args.model)
        if not default_model_path.exists():
            raise SystemExit(f"Model not found: {default_model_path}")
        default_model_id = default_model_path.name.rstrip(".onnx")
        _ = registry.ensure_loaded(default_model_id)
        registry.index.setdefault(default_model_id, default_model_path)

    # If no explicit default, pick the first discovered voice id (stable order)
    if (default_model_id is None) and registry.index:
        prefer = 'de_DE-thorsten-medium'
        default_model_id = prefer if prefer in registry.index else sorted(registry.index.keys())[0]

    app = Flask(__name__)
    sock = Sock(app)

    # Pipeline control API (requires --admin-token)
    admin_token = getattr(args, 'admin_token', None) or ''
    if admin_token:
        from speech_pipeline.pipeline_api import api as pipeline_api_bp, init as pipeline_api_init
        pipeline_api_init(admin_token)
        app.register_blueprint(pipeline_api_bp)
        _LOGGER.info("Pipeline control API enabled at /api/ (bearer-authenticated)")

        # Telephony API (shares the admin token, adds account-scoped auth)
        from speech_pipeline.telephony.auth import init as tel_auth_init
        from speech_pipeline.telephony.api import api as telephony_api_bp
        tel_auth_init(admin_token)
        app.register_blueprint(telephony_api_bp)
        _LOGGER.info("Telephony API enabled at /api/ (admin + account auth)")

        # Share TTS registry and app with telephony modules
        import speech_pipeline.telephony._shared as _tel_shared
        _tel_shared.tts_registry = registry
        _tel_shared.flask_app = app
        _tel_shared.media_folder = getattr(args, "media_folder", None)

        # WebClient: phone UI + WebSocket
        from speech_pipeline.telephony.webclient import bp as webclient_bp
        app.register_blueprint(webclient_bp)
        _LOGGER.info("WebClient phone UI enabled")

        # Startup callback: notify the external provisioner that the speech
        # server is ready.  In production this is LDS, but from the server's
        # perspective it is intentionally generic: the callback target is the
        # admin-side orchestrator that will provision PBXs/accounts via the
        # API.  The speech server itself does not know LDS-specific logic.
        startup_cb = getattr(args, 'startup_callback', None) or ''
        if startup_cb:
            startup_cb_token = getattr(args, 'startup_callback_token', None) or admin_token
            def _fire_startup_callback(url: str, token: str) -> None:
                import requests
                try:
                    _LOGGER.info("Sending startup callback to %s", url)
                    resp = requests.get(url, headers={
                        "Authorization": f"Bearer {token}",
                    }, timeout=30)
                    _LOGGER.info("Startup callback response: %s", resp.status_code)
                except Exception as e:
                    _LOGGER.warning("Startup callback failed: %s", e)
            # Fire after app is ready (in a thread to not block startup)
            threading.Thread(
                target=_fire_startup_callback,
                args=(startup_cb, startup_cb_token),
                daemon=True,
                name="startup-callback",
            ).start()

    # Ensure our module logger emits at desired level and propagates to root handler
    try:
        _LOGGER.setLevel(logging.DEBUG if args.debug else logging.INFO)
        _LOGGER.propagate = True
    except Exception:
        pass

    # (processing helpers removed; handled by stages/util.py)

    # (ffmpeg/ffprobe helpers removed)

    # Import pipeline stages
    try:
        # Allow importing from local lib/ directory
        import sys as _sys
        here = Path(__file__).resolve().parent
        _sys.path.insert(0, str(here / 'lib'))
        _sys.path.insert(0, str(here))
        from lib import AudioReader, VCConverter, PitchAdjuster, ResponseWriter, FileFetcher, RawResponseWriter  # type: ignore
    except Exception as _e:
        _LOGGER.warning('lib import failed: %s', _e)

    # (ffmpeg filter/run helpers removed)

    def _ffmpeg_pitch_shift(in_path: Path, out_path: Path, semitones: float, stop_check: Optional[Callable[[], bool]] = None) -> bool:
        if not _ffmpeg_exists():
            return False
        # factor >1 raises pitch; try asetrate + aresample + atempo to maintain duration
        factor = 2.0 ** (semitones / 12.0)
        # Prefer high-quality formant-preserving rubberband if available
        if _ffmpeg_has_filter('rubberband'):
            rb = f"rubberband=tempo=1.0:pitch={factor}:formant=1"
            cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(in_path), '-filter:a', rb, str(out_path)]
            _LOGGER.info("ffmpeg rubberband pitch: st=%.3f factor=%.5f cmd=%s", semitones, factor, ' '.join(cmd))
            if _run_ffmpeg_cmd(cmd, stop_check=stop_check):
                return True
            _LOGGER.warning("ffmpeg rubberband failed; falling back to asetrate/atempo")
        # Build chain of atempo filters to realize tempo=1/factor within 0.5..2.0 segments
        tempo = float(1.0 / factor)
        atempo_filters: List[str] = []
        if tempo <= 0:
            _LOGGER.warning("invalid tempo computed for pitch shift: %s", tempo)
            return False
        if tempo < 1.0:
            # Compose from 0.5 steps up to residual in [0.5, 1.0]
            remaining = tempo
            while remaining < 0.5:
                atempo_filters.append('atempo=0.5')
                remaining /= 0.5
            atempo_filters.append(f'atempo={remaining}')
        else:
            # Compose from 2.0 steps down to residual in [1.0, 2.0]
            remaining = tempo
            while remaining > 2.0:
                atempo_filters.append('atempo=2.0')
                remaining /= 2.0
            atempo_filters.append(f'atempo={remaining}')
        atempo_chain = ','.join(atempo_filters) if atempo_filters else ''
        # Use numeric input sample rate for stability across ffmpeg versions
        try:
            with _wave.open(str(in_path), 'rb') as _wf:
                in_sr = int(_wf.getframerate())
        except Exception:
            in_sr = None  # let ffmpeg infer; fall back to symbolic 'sample_rate' if available
        if in_sr and in_sr > 0:
            filt = f"asetrate={int(in_sr * factor)},aresample={in_sr}"
        else:
            # Fallback; most ffmpeg builds support 'sample_rate' variable in expressions
            filt = f"asetrate=sample_rate*{factor},aresample=sample_rate"
        if atempo_chain:
            filt = f"{filt},{atempo_chain}"
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(in_path), '-filter:a', filt, str(out_path)]
        _LOGGER.info("ffmpeg pitch: st=%.3f factor=%.5f tempo=%.5f cmd=%s", semitones, factor, tempo, ' '.join(cmd))
        return _run_ffmpeg_cmd(cmd, stop_check=stop_check)

    def _ffmpeg_change_speed(in_path: Path, out_path: Path, speed: float) -> bool:
        if not _ffmpeg_exists():
            return False
        spd = float(speed)
        if spd <= 0:
            _LOGGER.warning("invalid speed factor: %s", speed)
            return False
        # Build chain of atempo filters to realize any factor using 0.5..2.0 segments
        filters: List[str] = []
        remaining = spd
        if spd < 1.0:
            while remaining < 0.5:
                filters.append('atempo=0.5')
                remaining /= 0.5
            filters.append(f'atempo={remaining}')
        else:
            while remaining > 2.0:
                filters.append('atempo=2.0')
                remaining /= 2.0
            filters.append(f'atempo={remaining}')
        filt = ','.join(filters)
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(in_path), '-filter:a', filt, str(out_path)]
        try:
            _sp.run(cmd, check=True)
            return True
        except Exception as e:
            _LOGGER.warning("ffmpeg speed change failed: %s", e)
            return False

    def _ffmpeg_resample_mono_pad(in_path: Path, out_path: Path, sample_rate: int = 24000, pad_seconds: float = 0.2, stop_check: Optional[Callable[[], bool]] = None) -> bool:
        """Resample to mono at sample_rate and append short silence to avoid VC kernel-size errors."""
        if not _ffmpeg_exists():
            return False
        if sample_rate <= 0:
            sample_rate = 24000
        # apad pad_dur adds specified seconds of silence; keeps original audio
        filt = f"apad=pad_dur={max(0.0, float(pad_seconds))}"
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(in_path),
            '-ac', '1', '-ar', str(int(sample_rate)),
            '-filter:a', filt,
            '-c:a', 'pcm_s16le',
            str(out_path)
        ]
        _LOGGER.info("ffmpeg resample/mono: %s -> %s @ %d Hz", in_path, out_path, sample_rate)
        return _run_ffmpeg_cmd(cmd, stop_check=stop_check)

    def _ffmpeg_to_pcm16(in_path: Path, out_path: Path, sample_rate: Optional[int] = None, stop_check: Optional[Callable[[], bool]] = None) -> bool:
        """Force WAV PCM16 output (and optionally set sample rate)."""
        if not _ffmpeg_exists():
            return False
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(in_path), '-c:a', 'pcm_s16le']
        if sample_rate and sample_rate > 0:
            cmd += ['-ar', str(int(sample_rate))]
        cmd.append(str(out_path))
        _LOGGER.info("Converting to PCM16: %s -> %s%s", in_path, out_path, f" @{sample_rate}Hz" if sample_rate else "")
        return _run_ffmpeg_cmd(cmd, stop_check=stop_check)

    def _split_wav_pcm16(in_path: Path, chunk_seconds: float) -> List[Path]:
        """Split a PCM16 WAV into ~chunk_seconds chunks. Returns list of temp file paths."""
        import tempfile
        out_paths: List[Path] = []
        with _wave.open(str(in_path), 'rb') as wf:
            nchan = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            if nchan < 1 or sw != 2 or sr <= 0:
                raise RuntimeError('split_wav_pcm16: input must be PCM16 WAV')
            frames_per_chunk = int(max(1, sr * max(0.1, float(chunk_seconds))))
            while True:
                frames = wf.readframes(frames_per_chunk)
                if not frames:
                    break
                tmp = tempfile.NamedTemporaryFile(prefix='chunk_', suffix='.wav', delete=False)
                tmp_path = Path(tmp.name)
                tmp.close()
                with _wave.open(str(tmp_path), 'wb') as ww:
                    ww.setnchannels(nchan)
                    ww.setsampwidth(sw)
                    ww.setframerate(sr)
                    ww.writeframes(frames)
                out_paths.append(tmp_path)
        _LOGGER.info('split wav: %s -> %d chunks (~%.1fs each)', in_path, len(out_paths), chunk_seconds)
        return out_paths

    def _concat_wavs_pcm16(paths: List[Path], out_path: Path) -> bool:
        """Concatenate PCM16 WAVs into out_path."""
        if not paths:
            return False
        try:
            with _wave.open(str(paths[0]), 'rb') as wf0:
                nchan = wf0.getnchannels(); sw = wf0.getsampwidth(); sr = wf0.getframerate()
            with _wave.open(str(out_path), 'wb') as ww:
                ww.setnchannels(nchan); ww.setsampwidth(sw); ww.setframerate(sr)
                for p in paths:
                    with _wave.open(str(p), 'rb') as wf:
                        ww.writeframes(wf.readframes(wf.getnframes()))
            _LOGGER.info('concat wav: %d chunks -> %s', len(paths), out_path)
            return True
        except Exception as e:
            _LOGGER.warning('concat wav failed: %s', e)
            return False

    def _vc_convert_chunkwise(src_wav: Path, tgt_wav: Path, out_path: Path, chunk_seconds: float = CHUNKSIZE_SECONDS) -> bool:
        """Run FreeVC on src_wav in chunks and concatenate results to out_path (PCM16)."""
        import tempfile, os
        # Ensure inputs are PCM16 24k mono
        src_pcm = tempfile.NamedTemporaryFile(prefix='vc_src_pcm_', suffix='.wav', delete=False); src_pcm_path = Path(src_pcm.name); src_pcm.close()
        if not _ffmpeg_to_pcm16(src_wav, src_pcm_path, sample_rate=24000):
            _LOGGER.warning('vc chunkwise: src pcm16 conversion failed; using original')
            src_pcm_path = src_wav
        try:
            chunks = _split_wav_pcm16(src_pcm_path, chunk_seconds)
            out_chunks: List[Path] = []
            for i, c in enumerate(chunks):
                tmp = tempfile.NamedTemporaryFile(prefix=f'vc_chunk_{i:04d}_', suffix='.wav', delete=False)
                c_out = Path(tmp.name); tmp.close()
                try:
                    get_vc_model().voice_conversion_to_file(source_wav=str(c), target_wav=str(tgt_wav), file_path=str(c_out))  # type: ignore
                except Exception as e:
                    _LOGGER.warning('vc chunk %d failed: %s', i, e)
                    # Best-effort: pass through source chunk
                    c_out = c
                # Normalize each chunk to PCM16 24k for concatenation
                c_out_pcm = tempfile.NamedTemporaryFile(prefix=f'vc_chunk_pcm_{i:04d}_', suffix='.wav', delete=False)
                c_out_pcm_path = Path(c_out_pcm.name); c_out_pcm.close()
                if not _ffmpeg_to_pcm16(c_out, c_out_pcm_path, sample_rate=24000):
                    c_out_pcm_path = c_out
                out_chunks.append(c_out_pcm_path)
            ok = _concat_wavs_pcm16(out_chunks, out_path)
            return ok
        finally:
            try:
                if src_pcm_path != src_wav and src_pcm_path.exists():
                    os.unlink(src_pcm_path)
            except Exception:
                pass

    def _vc_pitch_concat_chunkwise(src_wav: Path, tgt_wav: Path, out_path: Path, semitones_override: Optional[float], chunk_seconds: float = CHUNKSIZE_SECONDS) -> bool:
        """VC + (optional) pitch per chunk, concatenate directly to out_path (PCM16@24k)."""
        import tempfile, os
        # Prepare source as PCM16 24k mono
        src_pcm = tempfile.NamedTemporaryFile(prefix='vc_src_pcm_', suffix='.wav', delete=False); src_pcm_path = Path(src_pcm.name); src_pcm.close()
        if not _ffmpeg_resample_mono_pad(src_wav, src_pcm_path, sample_rate=24000, pad_seconds=0.0):
            if not _ffmpeg_to_pcm16(src_wav, src_pcm_path, sample_rate=24000):
                src_pcm_path = src_wav
        # Estimate target baseline once (use full first chunk duration)
        try:
            sr_t, x_t = _read_wav_head_samples(tgt_wav, seconds=float(chunk_seconds))
            f0_t = _estimate_f0_avg(sr_t, x_t)
        except Exception:
            f0_t = None
        try:
            chunks = _split_wav_pcm16(src_pcm_path, chunk_seconds)
            # Open writer
            with _wave.open(str(out_path), 'wb') as ww:
                ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(24000)
                for i, c in enumerate(chunks):
                    # VC chunk
                    tmp_vc = tempfile.NamedTemporaryFile(prefix=f'vc_chunk_{i:04d}_', suffix='.wav', delete=False)
                    c_vc = Path(tmp_vc.name); tmp_vc.close()
                    try:
                        get_vc_model().voice_conversion_to_file(source_wav=str(c), target_wav=str(tgt_wav), file_path=str(c_vc))  # type: ignore
                    except Exception as e:
                        _LOGGER.warning('vc chunk %d failed: %s; passing through', i, e)
                        c_vc = c
                    # Ensure PCM16 24k for pitch/write
                    tmp_pcm = tempfile.NamedTemporaryFile(prefix=f'vc_chunk_pcm_{i:04d}_', suffix='.wav', delete=False)
                    c_pcm = Path(tmp_pcm.name); tmp_pcm.close()
                    if not _ffmpeg_to_pcm16(c_vc, c_pcm, sample_rate=24000):
                        c_pcm = c_vc
                    # Decide pitch for this chunk
                    applied_st: Optional[float] = None
                    if semitones_override is not None and abs(semitones_override) > 0.05:
                        applied_st = float(semitones_override) * float(PITCH_CORRECTION)
                    elif f0_t:
                        try:
                            sr_v, x_v = _read_wav_head_samples(c_pcm, seconds=float(chunk_seconds))
                            f0_v = _estimate_f0_avg(sr_v, x_v)
                            if f0_v and f0_v > 0.0:
                                st_raw = 12.0 * math.log2(float(f0_t) / float(f0_v))
                                applied_st = float(st_raw) * float(PITCH_CORRECTION)
                        except Exception:
                            applied_st = None
                    # Apply pitch if needed
                    c_out = c_pcm
                    if applied_st is not None and abs(applied_st) > 0.1:
                        tmp_ps = tempfile.NamedTemporaryFile(prefix=f'vc_chunk_ps_{i:04d}_', suffix='.wav', delete=False)
                        c_ps = Path(tmp_ps.name); tmp_ps.close()
                        if _ffmpeg_pitch_shift(c_pcm, c_ps, applied_st):
                            c_out = c_ps
                    # Append frames
                    try:
                        with _wave.open(str(c_out), 'rb') as wf:
                            ww.writeframes(wf.readframes(wf.getnframes()))
                    except Exception as e:
                        _LOGGER.warning('write chunk %d failed: %s (skipping)', i, e)
                    # cleanup temps (best-effort)
                    for p in (c_vc, c_pcm, c_out):
                        try:
                            if p not in (c,) and p.exists():
                                os.unlink(p)
                        except Exception:
                            pass
            return True
        finally:
            try:
                if src_pcm_path != src_wav and src_pcm_path.exists():
                    os.unlink(src_pcm_path)
            except Exception:
                pass


    def _apply_pitch_chunkwise(in_path: Path, out_path: Path, semitones: float, chunk_seconds: float = CHUNKSIZE_SECONDS) -> bool:
        """Split PCM16 WAV, pitch each chunk, and concatenate."""
        import tempfile, os
        # Ensure PCM16 for splitting
        in_pcm = tempfile.NamedTemporaryFile(prefix='pitch_in_pcm_', suffix='.wav', delete=False); in_pcm_path = Path(in_pcm.name); in_pcm.close()
        if not _ffmpeg_to_pcm16(in_path, in_pcm_path, None):
            in_pcm_path = in_path
        try:
            chunks = _split_wav_pcm16(in_pcm_path, chunk_seconds)
            out_chunks: List[Path] = []
            for i, c in enumerate(chunks):
                tmp = tempfile.NamedTemporaryFile(prefix=f'pitch_chunk_{i:04d}_', suffix='.wav', delete=False)
                c_out = Path(tmp.name); tmp.close()
                if not _ffmpeg_pitch_shift(c, c_out, semitones):
                    c_out = c
                out_chunks.append(c_out)
            ok = _concat_wavs_pcm16(out_chunks, out_path)
            return ok
        finally:
            try:
                if in_pcm_path != in_path and in_pcm_path.exists():
                    os.unlink(in_pcm_path)
            except Exception:
                pass

    def _python_resample_mono_pad(in_path: Path, out_path: Path, sample_rate: int = 24000, pad_seconds: float = 0.5) -> bool:
        """Pure-Python fallback: convert to mono, linear resample to sample_rate, pad trailing silence, write PCM16 WAV."""
        try:
            import wave as _w
            import numpy as _np
            with _w.open(str(in_path), 'rb') as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                n = wf.getnframes()
                raw = wf.readframes(n)
            if n == 0:
                # Create small silence
                sr = sample_rate
                x = _np.zeros(int(sr * max(0.5, pad_seconds)), dtype=_np.float32)
            else:
                if sw == 2:
                    x = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
                elif sw == 1:
                    x = (_np.frombuffer(raw, dtype=_np.uint8).astype(_np.float32) - 128.0) / 128.0
                else:
                    x = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
                if ch > 1:
                    x = x.reshape(-1, ch).mean(axis=1)
                # Resample if needed (linear)
                if sr != sample_rate and x.size > 1:
                    ratio = float(sample_rate) / float(sr)
                    out_len = max(1, int(round(x.size * ratio)))
                    t = _np.linspace(0.0, x.size - 1, num=out_len, dtype=_np.float32)
                    i0 = _np.floor(t).astype(_np.int32)
                    i1 = _np.minimum(i0 + 1, x.size - 1)
                    frac = t - i0
                    x = (x[i0] * (1.0 - frac)) + (x[i1] * frac)
                sr = sample_rate
                # Pad trailing silence
                pad = _np.zeros(int(max(0.0, pad_seconds) * sr), dtype=_np.float32)
                x = _np.concatenate([x, pad]) if pad.size > 0 else x
            # Write PCM16
            x16 = _np.clip(_np.round(x * 32767.0), -32768, 32767).astype(_np.int16)
            with _w.open(str(out_path), 'wb') as ww:
                ww.setnchannels(1)
                ww.setsampwidth(2)
                ww.setframerate(sr)
                ww.writeframes(x16.tobytes())
            _LOGGER.info("Python resample/mono/pad wrote: %s @ %d Hz (%d frames)", out_path, sr, x16.size)
            return True
        except Exception as e:
            _LOGGER.warning("python resample/mono failed: %s", e)
            return False

    # Basic CORS for all responses
    @app.after_request
    def add_cors_headers(resp):  # type: ignore
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
        # Allow Authorization so server-side bearer fetches are configurable if ever proxied
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization, X-Sample-Rate"
        # Media streaming hint for proxies
        resp.headers["X-Accel-Buffering"] = "no"
        # Be explicit about referrer policy if the browser surfaces strict-origin-when-cross-origin
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # Note: Content headers for audio are applied by ResponseWriter
        return resp

    # ---- Static files from examples/ directory ----
    from flask import send_from_directory as _send_from_directory
    _examples_dir = Path(__file__).resolve().parent / "examples"

    @app.route("/examples/<path:filename>")
    def serve_examples(filename):
        return _send_from_directory(str(_examples_dir), filename)

    # Operational endpoints (/healthz + /metrics) live in a shared
    # blueprint so the test app can exercise them too.
    from speech_pipeline.metrics_api import api as metrics_bp
    app.register_blueprint(metrics_bp)

    @app.route("/voices", methods=["GET"])  # list models
    def voices() -> Any:
        # Ensure index is current
        registry.refresh_index()
        result: Dict[str, Any] = {}
        for mid, path in registry.index.items():
            info = registry.infos.get(mid)
            if not info:
                try:
                    info = load_voice_info(path)
                    registry.infos[mid] = info
                except Exception:
                    info = None
            if info:
                result[mid] = {
                    "path": str(path),
                    "espeak_voice": info.espeak_voice,
                    "sample_rate": info.sample_rate,
                    "num_speakers": info.num_speakers,
                    "speaker_id_map": info.speaker_id_map,
                }
            else:
                result[mid] = {"path": str(path)}
        return jsonify(result)

    @app.route("/", methods=["OPTIONS"])  # CORS preflight
    @app.route("/tts/say", methods=["OPTIONS"])
    def options_root() -> Tuple[str, int, Dict[str, str]]:
        return ("", 204, {})

    # ---- Rate limiter for HTTP TTS (1 concurrent per real client IP) ----
    _http_tts_active: dict[str, int] = {}
    _http_tts_lock = threading.Lock()

    @app.route("/tts/say", methods=["GET", "POST"])
    @app.route("/", methods=["GET", "POST"])  # legacy compat
    def synthesize() -> Any:
        # Rate limit: 1 concurrent TTS request per IP
        client_ip = _get_real_ip()
        with _http_tts_lock:
            if _http_tts_active.get(client_ip, 0) >= 1:
                return ("Rate limit: only 1 concurrent TTS request per IP\n", 429)
            _http_tts_active[client_ip] = _http_tts_active.get(client_ip, 0) + 1
        try:
            return _synthesize_inner()
        finally:
            with _http_tts_lock:
                _http_tts_active[client_ip] = max(0, _http_tts_active.get(client_ip, 1) - 1)

    def _synthesize_inner() -> Any:
        # Accept JSON or form/query parameters for flexibility
        payload: Dict[str, Any] = {}
        if request.is_json:
            try:
                payload = request.get_json(force=True, silent=True) or {}
            except Exception:
                payload = {}
        # Merge form/query onto payload without overwriting explicit JSON values
        for k in ("text", "voice", "lang", "speaker", "speaker_id", "length_scale", "noise_scale", "noise_w_scale", "sentence_silence", "voice2", "sound", "pitch_st", "pitch_factor", "pitch_disable", "disable_pitch", "nopitch"):
            if k not in payload or payload.get(k) in (None, ""):
                v = request.form.get(k, request.args.get(k))
                if v is not None:
                    payload[k] = v
        text = (str(payload.get("text") or "")).strip()
        voice2 = (str(payload.get("voice2") or "")).strip()  # target VC reference (URL or media-folder filename)
        sound = (str(payload.get("sound") or "")).strip()    # source audio (URL or media-folder filename)
        # Require either text or sound; otherwise return generic 400 without hints
        if (not text) and (not sound):
            return ("bad request", 400, {"Content-Type": "text/plain"})
        # Optional pitch override controls (applied pre-VC)
        pitch_st_raw = (str(payload.get("pitch_st") or "").strip())
        pitch_factor_raw = (str(payload.get("pitch_factor") or "").strip())
        pitch_override_semitones: Optional[float] = None
        try:
            if pitch_st_raw != "":
                pitch_override_semitones = float(pitch_st_raw)
            elif pitch_factor_raw != "":
                pf = float(pitch_factor_raw)
                if pf > 0:
                    import math as _math
                    pitch_override_semitones = 12.0 * _math.log2(pf)
        except Exception:
            pitch_override_semitones = None
        # Quick switch to disable any pitch processing
        pitch_disable = bool(PITCH_DISABLE_DEFAULT)
        _pd = payload.get("pitch_disable", payload.get("disable_pitch", payload.get("nopitch")))
        try:
            if isinstance(_pd, bool):
                pitch_disable = _pd
            elif isinstance(_pd, (int, float)):
                pitch_disable = (float(_pd) != 0.0)
            elif isinstance(_pd, str):
                pitch_disable = _pd.strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            pitch_disable = bool(PITCH_DISABLE_DEFAULT)

        model_id = payload.get("voice") or default_model_id
        lang = payload.get("lang")
        # Resolve model by language if needed
        if not model_id and lang:
            model_id = registry.best_for_lang(lang)
        if not model_id and default_model_id:
            model_id = default_model_id
        if not model_id:
            return ("No voice available", 404, {"Content-Type": "text/plain"})

        # Validate model id by loading when needed through the factory
        try:
            _ = registry.ensure_loaded(model_id)
        except KeyError:
            return (f"Voice not found: {model_id}", 404, {"Content-Type": "text/plain"})

        try:
            sentence_silence = float(payload.get("sentence_silence", args.sentence_silence))
        except Exception:
            sentence_silence = args.sentence_silence
        _LOGGER.info("request: len(text)=%d model=%s voice2=%s sound=%s pitch_st=%s pitch_factor=%s pitch_disable=%s",
                     len(text), model_id, (voice2 or '-'), (sound or '-'),
                     payload.get('pitch_st'), payload.get('pitch_factor'), pitch_disable)

        from speech_pipeline.media_refs import resolve_media_ref

        def _resolve_media_arg(raw: str, field: str) -> str:
            try:
                return resolve_media_ref(raw, getattr(args, "media_folder", None))
            except ValueError as e:
                raise ValueError(f"{field}: {e}") from e

        # If 'sound' is provided: stream a WAV from voices/ optionally through VC to target 'voice2'
        if sound:
            try:
                value_s = _resolve_media_arg(sound, "sound")
            except ValueError as e:
                return (str(e), 400, {"Content-Type": "text/plain"})
            # If no voice2, shortcut via FileFetcher -> RawResponseWriter (no resample, raw passthrough)
            if not voice2:
                src_ref = value_s
                _LOGGER.info("Streaming sound (no VC) via FileFetcher: source=%s", src_ref)
                fetcher = FileFetcher(src_ref, bearer=getattr(args, 'bearer', ''))
                writer = RawResponseWriter(fetcher)
                # Guess a suitable mimetype from file extension
                guessed, _ = mimetypes.guess_type(src_ref)
                if guessed == 'audio/x-wav':
                    guessed = 'audio/wav'
                mtype = guessed or 'application/octet-stream'
                def gen_sound_only_raw():
                    for b in writer.stream():
                        yield b
                resp = Response(stream_with_context(gen_sound_only_raw()), mimetype=mtype)
                # Best-effort: set Content-Length if known to improve playback stability
                try:
                    if src_ref.startswith('http://') or src_ref.startswith('https://'):
                        try:
                            h = fetcher._open()  # type: ignore[attr-defined]
                            clen = getattr(h, 'getheader', None)
                            if callable(clen):
                                v = clen('Content-Length')
                                if v:
                                    resp.headers['Content-Length'] = str(v)
                            else:
                                v2 = getattr(getattr(h, 'headers', None), 'get', None)
                                if callable(v2):
                                    vv = v2('Content-Length')
                                    if vv:
                                        resp.headers['Content-Length'] = str(vv)
                        except Exception:
                            pass
                    else:
                        try:
                            resp.headers['Content-Length'] = str(os.path.getsize(src_ref))
                        except Exception:
                            pass
                except Exception:
                    pass
                # Ensure inline disposition for browser playback
                try:
                    resp.headers.setdefault('Content-Disposition', 'inline')
                except Exception:
                    pass
                def _cleanup():
                    try:
                        writer.cancel()
                    except Exception:
                        pass
                    try:
                        fetcher.close()
                    except Exception:
                        pass
                resp.call_on_close(_cleanup)
                return resp

            # With voice2: convert whole file using VC stage (passes through if unavailable)
            try:
                value_t = _resolve_media_arg(voice2, "voice2")
            except ValueError as e:
                return (str(e), 400, {"Content-Type": "text/plain"})
            src_ref = value_s
            _LOGGER.info("SOUND+VC: source ref=%s target ref=%s (downloading if http)", src_ref, value_t)
            source = AudioReader(src_ref, bearer=getattr(args, 'bearer', ''))
            # Let stages resolve and fetch target as needed (with bearer), avoiding temp logic here
            pipeline = source.pipe(VCConverter(value_t, bearer=getattr(args, 'bearer', ''))).pipe(PitchAdjuster(value_t, pitch_disable=False, pitch_override_st=None, correction=PITCH_CORRECTION, bearer=getattr(args, 'bearer', '')))
            writer = ResponseWriter(pipeline, est_frames_24k=source.estimate_frames_24k())
            def gen_stream_sound():
                for b in writer.stream():
                    yield b
            resp = Response(stream_with_context(gen_stream_sound()), mimetype="audio/wav")
            try:
                writer.apply_headers(resp)
            except Exception:
                pass
            def _cleanup2():
                try:
                    writer.cancel()
                except Exception:
                    pass
            resp.call_on_close(_cleanup2)
            return resp

        # TTS path: always stream via pipeline (realtime)
        # If voice2 is provided, run VC + pitch; else just TTS

        # TTS+VC pipeline (still realtime streaming)
        _LOGGER.info("path: TTS pipeline (VC=%s)", bool(voice2))

        # 2) If voice2 requested, run VC; stage handles passthrough if VC unavailable
        if voice2:
            try:
                value_t = _resolve_media_arg(voice2, "voice2")
            except ValueError as e:
                return (str(e), 400, {"Content-Type": "text/plain"})
            # Build pipeline: TTSProducer -> VC -> Pitch -> Writer
            _LOGGER.info("TTS+VC: target ref=%s (downloading if http)", value_t)
            source = registry.create_tts_stream(model_id, text, {"sentence_silence": sentence_silence, "chunk_seconds": CHUNKSIZE_SECONDS, "speaker": payload.get("speaker"), "speaker_id": payload.get("speaker_id"), "length_scale": payload.get("length_scale"), "noise_scale": payload.get("noise_scale"), "noise_w_scale": payload.get("noise_w_scale")} )
            # Let stages resolve and fetch target as needed (with bearer)
            pipeline = source.pipe(VCConverter(value_t, bearer=getattr(args, 'bearer', ''))).pipe(PitchAdjuster(value_t, pitch_disable, pitch_override_semitones, correction=PITCH_CORRECTION, bearer=getattr(args, 'bearer', '')))
            writer = ResponseWriter(pipeline, est_frames_24k=source.estimate_frames_24k())
            def gen_stream():
                for b in writer.stream():
                    yield b
            resp = Response(stream_with_context(gen_stream()), mimetype="audio/wav")
            try:
                writer.apply_headers(resp)
            except Exception:
                pass
            def _cleanup3():
                try:
                    writer.cancel()
                except Exception:
                    pass
            resp.call_on_close(_cleanup3)
            return resp

        # 3) No VC: stream TTS via pipeline (single header + PCM chunks)
        _LOGGER.info("path: buffered TTS (no VC) via pipeline")
        source = registry.create_tts_stream(model_id, text, {"sentence_silence": sentence_silence, "chunk_seconds": CHUNKSIZE_SECONDS, "speaker": payload.get("speaker"), "speaker_id": payload.get("speaker_id"), "length_scale": payload.get("length_scale"), "noise_scale": payload.get("noise_scale"), "noise_w_scale": payload.get("noise_w_scale")} )
        writer = ResponseWriter(source, est_frames_24k=source.estimate_frames_24k())
        def gen_tts_only():
            for b in writer.stream():
                yield b
        resp = Response(stream_with_context(gen_tts_only()), mimetype="audio/wav")
        try:
            writer.apply_headers(resp)
        except Exception:
            pass
        resp.call_on_close(lambda: writer.cancel())
        return resp

    # /inputstream removed — use /ws/stt or POST /api/pipelines instead

    # ---- Streaming TTS: params via GET, POST body = text stream, response = audio stream ----
    @app.route("/tts/stream", methods=["POST", "PUT", "OPTIONS"])
    def tts_stream():
        if request.method == "OPTIONS":
            return ("", 204)
        # Rate limit: 1 concurrent per IP (same as /tts/say)
        client_ip = _get_real_ip()
        with _http_tts_lock:
            if _http_tts_active.get(client_ip, 0) >= 1:
                return ("Rate limit: only 1 concurrent TTS request per IP\n", 429)
            _http_tts_active[client_ip] = _http_tts_active.get(client_ip, 0) + 1
        try:
            return _tts_stream_inner()
        finally:
            with _http_tts_lock:
                _http_tts_active[client_ip] = max(0, _http_tts_active.get(client_ip, 1) - 1)

    def _tts_stream_inner():
        model_id = request.args.get("voice", default_model_id)
        voice = registry.ensure_loaded(model_id)
        syn = registry.create_synthesis_config(voice, request.args.to_dict())

        def text_lines():
            """Yield text lines from the streamed POST/PUT body."""
            import os
            stream = request.environ['wsgi.input']
            # os.read() on the fd: one syscall, returns all available bytes
            # immediately (up to 4096), blocks only when nothing is available.
            try:
                fd = stream.fileno()
                use_fd = True
            except Exception:
                use_fd = False
            buf = b""
            while True:
                chunk = os.read(fd, 4096) if use_fd else stream.read(1)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        yield text
            text = buf.decode("utf-8", errors="replace").strip()
            if text:
                yield text

        # Pipeline: StreamingTTSProducer -> ResponseWriter
        # max_chunk_bytes=4800 → ~0.1s chunks at 24kHz/16bit for low-latency playback
        from lib import StreamingTTSProducer, ResponseWriter
        source = StreamingTTSProducer(text_lines(), voice, syn)
        writer = ResponseWriter(source, est_frames_24k=0x3FFFFFFF, max_chunk_bytes=4800)

        resp = Response(stream_with_context(writer.stream()), mimetype="audio/wav")
        resp.headers["X-Accel-Buffering"] = "no"
        resp.headers["Cache-Control"] = "no-cache"
        # No Content-Length for streaming — forces chunked transfer, prevents client buffering
        resp.call_on_close(lambda: source.cancel())
        return resp

    # ---- Real IP detection (for rate limiting behind reverse proxy) ----
    import threading as _ws_threading

    def _get_real_ip() -> str:
        """Get real client IP behind reverse proxy (X-Forwarded-For)."""
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.remote_addr or "unknown"

    # ---- WebSocket concurrency limiter (1 per IP for STT/TTS) ----
    _ws_active: dict[str, int] = {}   # ip -> count
    _ws_active_lock = _ws_threading.Lock()

    class _WSConcurrencyGuard:
        """Context manager: limits concurrent WS connections per IP."""
        def __init__(self, ip: str, limit: int = 1):
            self.ip = ip
            self.limit = limit
            self.acquired = False

        def __enter__(self):
            with _ws_active_lock:
                current = _ws_active.get(self.ip, 0)
                if current >= self.limit:
                    return False
                _ws_active[self.ip] = current + 1
                self.acquired = True
                return True

        def __exit__(self, *exc):
            if self.acquired:
                with _ws_active_lock:
                    _ws_active[self.ip] = max(0, _ws_active.get(self.ip, 1) - 1)

    def _ws_require_auth(ws) -> dict | None:
        """Check Bearer token from query param or first WS message.
        Returns account dict (or None for admin). Sends error and returns False on failure."""
        token = request.args.get("token", "")
        if not token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
        if not token:
            return False
        if not admin_token:
            return False
        if token == admin_token:
            return None  # admin
        from speech_pipeline.telephony.auth import _find_account_by_token
        acct = _find_account_by_token(token)
        if acct:
            return acct
        # Webclient nonces (``n-...``) are short-lived per-session
        # tokens that the CRM-issued iframe embeds.  They authorize
        # exactly one webclient conference pipe.  Accept them here so
        # phone.html can open /ws/pipe without handing the browser a
        # full account token.
        if token.startswith("n-"):
            from speech_pipeline.telephony.auth import check_nonce
            if check_nonce(token):
                return {"id": "__webclient__", "nonce": token}
        return False

    # ---- WebSocket STT streaming endpoint ----
    @sock.route("/ws/stt")
    def ws_stt(ws):
        """WebSocket STT: client sends codec audio, server replies with NDJSON text.

        Protocol:
        - Client → Server: hello handshake, then binary codec frames
        - Server → Client: hello response, then text NDJSON lines, "__END__" when done

        Rate limit: 1 concurrent connection per IP.
        """
        client_ip = _get_real_ip()
        guard = _WSConcurrencyGuard(client_ip, limit=1)
        with guard as allowed:
            if not allowed:
                import json as _json
                ws.send(_json.dumps({"error": "Rate limit: only 1 concurrent STT session per IP"}))
                return
            _ws_stt_inner(ws)

    def _ws_stt_inner(ws):
        from lib import SampleRateConverter
        from lib.WhisperSTT import WhisperTranscriber
        from lib import fourier_codec as codec
        import json as _json

        model_size = getattr(args, 'whisper_model', 'small')

        # Codec handshake
        try:
            msg = ws.receive(timeout=10)
        except Exception:
            msg = None

        profile = "low"
        language = None
        if isinstance(msg, str):
            try:
                obj = _json.loads(msg)
                if obj.get("type") == "hello":
                    client_profiles = [p for p in obj.get("profiles", []) if p in codec.PROFILES]
                    if client_profiles:
                        profile = client_profiles[0]
                    lang = obj.get("language")
                    if lang and isinstance(lang, str) and len(lang) <= 5:
                        language = lang.strip()
            except Exception:
                pass

        ws.send(_json.dumps({"type": "hello", "profile": profile}))

        # Inline source: decode codec frames from WS into PCM
        class _CodecWsSource:
            def __init__(self):
                self.cancelled = False
            def stream_pcm24k(self):
                while not self.cancelled:
                    try:
                        msg = ws.receive(timeout=60)
                    except Exception:
                        break
                    if msg is None:
                        break
                    if isinstance(msg, str):
                        if msg.strip() == "__END__":
                            break
                        continue
                    try:
                        samples, _prof = codec.decode_frame(msg)
                        yield codec.float32_to_pcm_s16le(samples)
                    except Exception as e:
                        _LOGGER.debug("ws_stt decode error: %s", e)

        src = _CodecWsSource()
        stt = WhisperTranscriber(model_size, chunk_seconds=2.0, language=language)
        pipeline = SampleRateConverter(48000, 16000)
        pipeline.set_upstream(src)
        stt.set_upstream(pipeline)

        try:
            for chunk in stt.stream_pcm24k():
                if src.cancelled:
                    break
                for line in chunk.decode('utf-8', errors='replace').splitlines():
                    line = line.strip()
                    if line:
                        ws.send(line)
        except Exception as e:
            _LOGGER.warning("ws_stt error: %s", e)
        finally:
            try:
                ws.send("__END__")
            except Exception:
                pass

    # ---- WebSocket TTS streaming endpoint ----
    @sock.route("/ws/tts")
    def ws_tts(ws):
        """WebSocket TTS: client sends text, server replies with codec audio.

        Protocol:
        - Client → Server: hello handshake, then text messages, "__END__" to signal done
        - Server → Client: hello response, then binary codec frames, "__END__" when done
        - Voice via query param: ?voice=de_DE-thorsten-medium

        Rate limit: 1 concurrent connection per IP.
        """
        client_ip = _get_real_ip()
        guard = _WSConcurrencyGuard(client_ip, limit=1)
        with guard as allowed:
            if not allowed:
                import json as _json
                ws.send(_json.dumps({"error": "Rate limit: only 1 concurrent TTS session per IP"}))
                return
            _ws_tts_inner(ws)

    def _ws_tts_inner(ws):
        from lib import StreamingTTSProducer, SampleRateConverter
        from lib import fourier_codec as codec
        import json as _json

        model_id = request.args.get("voice", default_model_id)
        try:
            voice = registry.ensure_loaded(model_id)
        except KeyError:
            ws.send(_json.dumps({"error": f"Voice not found: {model_id}"}))
            return
        syn = registry.create_synthesis_config(voice, request.args.to_dict())
        sample_rate = voice.config.sample_rate

        # Codec handshake
        try:
            msg = ws.receive(timeout=10)
        except Exception:
            msg = None

        profile = "low"
        if isinstance(msg, str):
            try:
                obj = _json.loads(msg)
                if obj.get("type") == "hello":
                    client_profiles = [p for p in obj.get("profiles", []) if p in codec.PROFILES]
                    if client_profiles:
                        profile = client_profiles[0]
            except Exception:
                pass

        ws.send(_json.dumps({"type": "hello", "profile": profile}))

        # Read text from WS, produce TTS, encode with codec, send back
        def _text_lines():
            while True:
                try:
                    msg = ws.receive(timeout=60)
                except Exception:
                    break
                if msg is None:
                    break
                if isinstance(msg, bytes):
                    continue
                text = msg.strip()
                if text == "__END__":
                    break
                if text:
                    yield text

        source = StreamingTTSProducer(_text_lines(), voice, syn)
        pipeline = source
        if sample_rate != 48000:
            pipeline = pipeline.pipe(SampleRateConverter(sample_rate, 48000))

        frame_bytes = codec.FRAME_SAMPLES * 2  # s16le
        buf = b""
        try:
            for pcm in pipeline.stream_pcm24k():
                buf += pcm
                while len(buf) >= frame_bytes:
                    chunk = buf[:frame_bytes]
                    buf = buf[frame_bytes:]
                    samples = codec.pcm_s16le_to_float32(chunk)
                    encoded = codec.encode_frame(samples, profile)
                    ws.send(encoded)
            if buf:
                padded = buf + b"\x00" * (frame_bytes - len(buf))
                samples = codec.pcm_s16le_to_float32(padded)
                encoded = codec.encode_frame(samples, profile)
                ws.send(encoded)
        except Exception as e:
            _LOGGER.warning("ws_tts error: %s", e)
        finally:
            try:
                ws.send("__END__")
            except Exception:
                pass

    # ---- Generic pipeline endpoint ----
    @sock.route("/ws/pipe")
    def ws_pipe(ws):
        """Generic pipeline endpoint: client sends JSON config, then data flows.

        Requires authentication: pass token as ?token=... query param
        or Authorization: Bearer header.

        Protocol:
        - Client -> Server: first message is JSON config:
            {"pipe": "ws:pcm | resample:48000:16000 | stt:de | ws:ndjson"}
          or for duplex:
            {"pipes": ["sip:100@pbx | resample:8000:16000 | stt:de | ws:ndjson",
                        "ws:text | tts{\"voice\":\"de_DE-thorsten-medium\"} | resample:24000:8000 | sip:100@pbx"]}
        - Then data flows according to the pipeline definition.
        """
        import json as _json
        if admin_token:
            auth_result = _ws_require_auth(ws)
            if auth_result is False:
                ws.send(_json.dumps({"error": "Unauthorized — pass ?token=... or Authorization header"}))
                return

        from lib.PipelineBuilder import PipelineBuilder

        try:
            config_msg = ws.receive(timeout=10)
        except Exception:
            ws.send(_json.dumps({"error": "No config message received"}))
            return

        try:
            config = _json.loads(config_msg)
        except Exception:
            ws.send(_json.dumps({"error": "Invalid JSON config"}))
            return

        # Track in LivePipeline registry if admin API is enabled
        from speech_pipeline.live_pipeline import LivePipeline, register as _lp_register, unregister as _lp_unregister
        dsl = config.get("pipe", "") or "; ".join(config.get("pipes", []))
        lp = LivePipeline(dsl=dsl)
        builder = PipelineBuilder(ws, registry, args, live_pipeline=lp)

        # Inject conference mixers referenced in the DSL
        from speech_pipeline.PipelineBuilder import inject_conference_mixers
        inject_conference_mixers(builder, dsl)

        # After build, wire ConferenceLeg callbacks to subscriber events
        def _wire_leg_callbacks(run_obj):
            _LOGGER.info("_wire_leg_callbacks called with %d stages", len(run_obj.stages))
            try:
                from speech_pipeline.ConferenceLeg import ConferenceLeg
                from speech_pipeline.telephony.webclient import get_webclient_session
                from speech_pipeline.telephony import call_state as _cs, subscriber as _sub
                import requests as _http
                import re as _re

                leg_count = sum(1 for s in run_obj.stages if isinstance(s, ConferenceLeg))
                _LOGGER.info("_wire_leg_callbacks: %d legs in %d stages, dsl=%s",
                             leg_count, len(run_obj.stages), dsl[:80])
                for stage in run_obj.stages:
                    if not isinstance(stage, ConferenceLeg):
                        continue
                    # Find the webclient session for this leg
                    for sid_match in _re.findall(r'codec:(wc-[^\s|]+)', dsl):
                        sess = get_webclient_session(sid_match)
                        if not sess:
                            continue
                        call = _cs.get_call(sess["call_id"])
                        if not call:
                            continue
                        sub = _sub.get(call.subscriber_id)
                        if not sub:
                            continue
                        participant = call.get_participant(sess["session_id"])
                        callback_path = participant.get("callback") if participant else None

                        _LOGGER.info("_wire: participant=%s callback=%s", sess["session_id"], callback_path)

                        def _make_on_attached(sub_info, call_obj, cb_path, part_id):
                            def _on_attached(leg):
                                _LOGGER.info("on_attached fired for %s cb=%s", part_id, cb_path)
                                if not cb_path:
                                    return
                                url = sub_info["base_url"].rstrip("/") + "/" + cb_path.lstrip("/")
                                # MUST be async — this fires from the
                                # mixer's yield loop; a blocking POST
                                # deadlocks the whole conference until
                                # the CRM responds.
                                def _send():
                                    try:
                                        _http.post(url, json={
                                            "callId": call_obj.call_id,
                                            "command": "webclient",
                                            "participantId": part_id,
                                            "result": "joined",
                                        }, headers={"Authorization": f"Bearer {sub_info['bearer_token']}"}, timeout=5)
                                    except Exception:
                                        pass
                                threading.Thread(
                                    target=_send, daemon=True,
                                    name=f"wc-attach-{part_id}",
                                ).start()
                            return _on_attached

                        stage.on_attached = _make_on_attached(sub, call, callback_path, sess["session_id"])

                        def _make_on_detached(sub_info, call_obj, cb_path, part_id):
                            def _on_detached(leg):
                                if not cb_path:
                                    return
                                url = sub_info["base_url"].rstrip("/") + "/" + cb_path.lstrip("/")
                                def _send():
                                    try:
                                        _http.post(url, json={
                                            "callId": call_obj.call_id,
                                            "command": "webclient",
                                            "participantId": part_id,
                                            "result": "left",
                                        }, headers={"Authorization": f"Bearer {sub_info['bearer_token']}"}, timeout=5)
                                    except Exception:
                                        pass
                                threading.Thread(
                                    target=_send, daemon=True,
                                    name=f"wc-detach-{part_id}",
                                ).start()
                            return _on_detached

                        stage.on_detached = _make_on_detached(sub, call, callback_path, sess["session_id"])
            except ImportError as e:
                _LOGGER.warning("_wire_leg_callbacks import error: %s", e)
            except Exception as e:
                _LOGGER.warning("_wire_leg_callbacks error: %s", e, exc_info=True)

        try:
            if "pipe" in config:
                run = builder.build(config["pipe"])
                _wire_leg_callbacks(run)
                lp.run = run
                lp.state = "running"
                _lp_register(lp)
                run.run()
            elif "pipes" in config:
                runs = builder.build_multi(config["pipes"])
                for r in runs:
                    _wire_leg_callbacks(r)
                lp.run = runs[0] if runs else None
                lp.state = "running"
                _lp_register(lp)
                if len(runs) == 1:
                    runs[0].run()
                else:
                    threads = [threading.Thread(target=r.run, daemon=True) for r in runs]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
            else:
                ws.send(_json.dumps({"error": "Config must contain 'pipe' or 'pipes'"}))
        except Exception as e:
            _LOGGER.warning("ws_pipe error: %s", e, exc_info=True)
            try:
                ws.send(_json.dumps({"error": str(e)}))
            except Exception:
                pass
        finally:
            lp.state = "stopped"
            _lp_unregister(lp.id)

    # ---- Codec WebSocket endpoint ----
    @sock.route("/ws/socket/<session_id>")
    def ws_codec_socket(ws, session_id):
        """Fourier-codec bidirectional audio socket.

        Pipeline stages create a CodecSocketSession for *session_id*.
        The browser connects here; the session handles handshake +
        encode/decode and bridges to the pipeline via rx/tx queues.
        """
        from lib.CodecSocketSession import get_session
        from speech_pipeline.CodecSocketSession import CodecSocketSession
        from speech_pipeline.telephony.webclient import get_webclient_session
        import json as _json

        # Browser may open /ws/socket/<sid> in parallel with /ws/pipe
        # — there's no guarantee the pipeline has registered the
        # CodecSocketSession yet when we get here.  Pre-create the
        # session if it doesn't exist but the webclient slot is known.
        # PipelineBuilder's _get_or_create_codec_session will reuse it.
        session = get_session(session_id)
        if not session:
            # Webclient session IDs are now account-scoped ("account:wc-...").
            # Pre-create a codec session for every known webclient slot,
            # regardless of whether the ID is scoped or legacy-local.
            if get_webclient_session(session_id):
                session = CodecSocketSession(session_id)
                _LOGGER.info(
                    "ws_codec_socket: pre-created CodecSocketSession for %s",
                    session_id,
                )
            else:
                ws.send(_json.dumps({
                    "error": "Unknown session ID",
                    "session_id": session_id,
                }))
                return
        session.handle_ws(ws)
        # NB: don't close_webclient_session here — browsers do a normal
        # disconnect-reconnect cycle on page load, and tearing down the
        # session on the first WS end makes every reconnect fail.
        # Cleanup happens on:
        #   - explicit DELETE /api/calls/<id>  → close_call_sessions
        #   - reaper thread for never-connected sessions (5 min TTL)
        #   - call idle-timeout (ConferenceMixer.on_idle_cancel)

    # ---- WebSocket TTS streaming endpoint (alternative path) ----
    @sock.route("/tts/ws")
    def tts_ws(ws):
        """WebSocket TTS: client sends text lines, server replies with PCM audio.

        Protocol:
        - Client → Server: text messages (sentences), "__END__" to signal done
        - Server → Client: text '{"sample_rate":24000}' once, then binary PCM
          s16le mono chunks, text "__END__" when done.
        - Voice via query param: ?voice=de_DE-thorsten-medium
        """
        from lib import WebSocketReader, WebSocketWriter, StreamingTTSProducer
        import json as _json

        model_id = request.args.get("voice", default_model_id)
        try:
            voice = registry.ensure_loaded(model_id)
        except KeyError:
            ws.send(_json.dumps({"error": f"Voice not found: {model_id}"}))
            return
        syn = registry.create_synthesis_config(voice, request.args.to_dict())
        sample_rate = voice.config.sample_rate

        # Send config message
        ws.send(_json.dumps({"sample_rate": sample_rate}))

        reader = WebSocketReader(ws)
        source = StreamingTTSProducer(reader.text_lines(), voice, syn)
        writer = WebSocketWriter(ws, source, max_chunk_bytes=4800)
        writer.run()

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", help="Optional: preload this voice (.onnx path)")
    parser.add_argument("--voices-path", default="voices-piper", help="Directory that contains *.onnx voices (default: voices-piper)")
    parser.add_argument("--scan-dir", help="(legacy) Single directory to scan for *.onnx voices; same as --voices-path")
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument("--sentence-silence", type=float, default=0.0, help="Seconds of silence between sentences")
    parser.add_argument("--media-folder", default=None, help="Folder for relative play/vc media names. URLs remain allowed; '../' is rejected.")
    parser.add_argument("--soundpath", default="../voices/%s.wav", help="Deprecated voice-ref template. Prefer JSON params with URLs or --media-folder filenames.")
    parser.add_argument("--bearer", default="", help="Bearer token for authorizing remote (http/https) downloads/streams")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size for STT (default: base)")
    parser.add_argument("--admin-token", default="", help="Bearer token to enable the /api/ pipeline control endpoints")
    parser.add_argument("--startup-callback", default="", help="URL to GET on startup. The called service provisions PBX/accounts via the API.")
    parser.add_argument("--startup-callback-token", default="", help="Bearer token for the startup callback (if different from --admin-token).")
    parser.add_argument("--sip-port", type=int, default=0, help="Enable built-in SIP stack on this UDP port (e.g. 5061). Disables pyVoIP.")
    parser.add_argument(
        "--telephony-log-level",
        type=int,
        default=2,
        choices=range(0, 5),
        help="Telephony verbosity: 0=off, 1=register/provision/init, 2=calls, 3=webhooks, 4=all SIP/debug.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        from speech_pipeline.telephony import logcontrol as _tel_logcontrol
        effective_tel_level = _tel_logcontrol.configure(args.telephony_log_level, debug=args.debug)
        _LOGGER.info("Telephony log level set to %d", effective_tel_level)
    except Exception:
        _LOGGER.warning("Failed to configure telephony log level", exc_info=True)

    # Fix pyVoIP trans() timing (must be set before any calls)
    try:
        import pyVoIP
        pyVoIP.TRANSMIT_DELAY_REDUCTION = 1.0
    except ImportError:
        pass

    # Start built-in SIP stack if requested
    if args.sip_port:
        from speech_pipeline.telephony import sip_stack
        sip_stack.init(args.sip_port)
        _LOGGER.info("Built-in SIP stack enabled on port %d", args.sip_port)

    app = create_app(args)

    # Graceful shutdown: on SIGTERM/SIGINT, hang up active legs + close
    # webclient sessions so SIP/WebSocket peers see a clean BYE/close
    # instead of a dead socket.
    import signal as _signal

    def _graceful_shutdown(signum, _frame):
        _LOGGER.info("Signal %d received — draining active calls", signum)
        try:
            from speech_pipeline.telephony import call_state as _cs
            for call_id in list(_cs._calls.keys()):
                try:
                    _cs.delete_call(call_id)
                except Exception as e:
                    _LOGGER.warning("shutdown: delete_call(%s) failed: %s",
                                    call_id, e)
        except Exception as e:
            _LOGGER.warning("shutdown drain error: %s", e)
        # Re-raise default behaviour so the process exits.
        _signal.signal(signum, _signal.SIG_DFL)
        import os
        os.kill(os.getpid(), signum)

    _signal.signal(_signal.SIGTERM, _graceful_shutdown)
    _signal.signal(_signal.SIGINT, _graceful_shutdown)

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
