from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Iterator, List, Optional

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("whisper-stt")

try:
    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore
except Exception:
    _WhisperModel = None  # type: ignore

_singleton_model = None
_singleton_lock = threading.Lock()
_model_init_lock = threading.Lock()
_INGEST_QUEUE_MAXSIZE = int(os.environ.get("WHISPER_INGEST_QUEUE_MAXSIZE", "3000"))
_INGEST_LOG_INTERVAL = 5.0
_SILENCE_RMS_FLOOR = int(os.environ.get("WHISPER_SILENCE_RMS_FLOOR", "220"))
_VOICE_FRAME_RMS = int(os.environ.get("WHISPER_VOICE_FRAME_RMS", "420"))
_VOICE_FRAME_MS = 20
_MIN_VOICED_MS = int(os.environ.get("WHISPER_MIN_VOICED_MS", "120"))
_MIN_VOICED_RATIO = float(os.environ.get("WHISPER_MIN_VOICED_RATIO", "0.08"))


def _detect_device() -> str:
    device = os.environ.get("WHISPER_DEVICE", "").lower()
    if device in ("cuda", "cpu"):
        return device
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _device_candidates(preferred):
    if preferred == "cuda":
        yield "cuda", "float16"
        yield "cuda", "int8"
    yield "cpu", "int8"
    yield "cpu", "float32"


def _get_model(model_size: str = "small"):
    """Return a process-wide singleton WhisperModel, lazily loaded."""
    global _singleton_model
    if _singleton_model is not None:
        return _singleton_model
    with _model_init_lock:
        if _singleton_model is not None:
            return _singleton_model
        if _WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed")
        device = _detect_device()
        for dev, ct in _device_candidates(device):
            try:
                _LOGGER.info("Loading Whisper model %s on %s (compute_type=%s)",
                             model_size, dev, ct)
                _singleton_model = _WhisperModel(model_size, device=dev,
                                                  compute_type=ct)
                return _singleton_model
            except (ValueError, RuntimeError) as e:
                _LOGGER.warning("Whisper init failed (%s/%s): %s", dev, ct, e)
        raise RuntimeError("Could not load Whisper model on any device")


class WhisperTranscriber(Stage):
    """Sink stage: consumes PCM s16le from upstream, yields NDJSON lines.

    Buffers ~chunk_seconds of audio, transcribes via faster-whisper,
    and yields one JSON line per recognized segment.
    """

    def __init__(self, model_size: str = "small", chunk_seconds: float = 3.0,
                 sample_rate: int = 16000, language: Optional[str] = None) -> None:
        super().__init__()
        self.model_size = model_size
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate
        self.language = language
        self.input_format = AudioFormat(sample_rate, "s16le")
        self.output_format = AudioFormat(0, "ndjson")

    def ensure_model_loaded(self) -> None:
        """Pre-load the Whisper model so stream_pcm24k() starts instantly."""
        _get_model(self.model_size)

    def stream_pcm24k(self) -> Iterator[bytes]:
        """Yields NDJSON lines (as bytes) instead of PCM.

        Accumulates audio and transcribes at natural pause boundaries
        (silence detection) rather than fixed time intervals, so words
        are never split mid-utterance.
        """
        import numpy as np
        if not self.upstream:
            return

        pcm_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=_INGEST_QUEUE_MAXSIZE)
        ingest_done = threading.Event()
        stop_ingest = threading.Event()
        dropped_chunks = 0
        last_drop_log = 0.0

        def _ingest() -> None:
            nonlocal dropped_chunks, last_drop_log
            try:
                for pcm in self.upstream.stream_pcm24k():
                    if self.cancelled or stop_ingest.is_set():
                        break
                    try:
                        pcm_queue.put_nowait(pcm)
                    except queue.Full:
                        dropped_chunks += 1
                        now = time.monotonic()
                        if now - last_drop_log >= _INGEST_LOG_INTERVAL:
                            _LOGGER.warning(
                                "Whisper ingest queue full, dropped %d chunks in %.1fs",
                                dropped_chunks,
                                _INGEST_LOG_INTERVAL,
                            )
                            dropped_chunks = 0
                            last_drop_log = now
            finally:
                ingest_done.set()
                try:
                    pcm_queue.put_nowait(None)
                except queue.Full:
                    pass

        ingest_thread = threading.Thread(
            target=_ingest,
            daemon=True,
            name="whisper-ingest",
        )
        ingest_thread.start()

        model = _get_model(self.model_size)

        bps = self.sample_rate * 2  # bytes per second (s16le)
        min_chunk_bytes = int(bps * 1.0)     # at least 1s before transcribing
        max_chunk_bytes = int(bps * 15.0)    # safety net: transcribe after 15s
        silence_trigger = int(bps * 0.3)     # 300ms silence = pause detected
        rms_floor = _SILENCE_RMS_FLOOR        # int16 RMS below this = silence

        buf = b""
        time_offset = 0.0
        silence_run = 0

        _LOGGER.info(
            "WhisperTranscriber: pause-based chunking (silence=300ms, min=1s, max=15s, queue=%d)",
            _INGEST_QUEUE_MAXSIZE,
        )
        try:
            while not self.cancelled:
                try:
                    pcm = pcm_queue.get(timeout=0.5)
                except queue.Empty:
                    if ingest_done.is_set():
                        break
                    continue
                if pcm is None:
                    break

                # Track silence
                samples = np.frombuffer(pcm, dtype=np.int16)
                rms = int(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                if rms < rms_floor:
                    silence_run += len(pcm)
                else:
                    silence_run = 0

                buf += pcm

                # Transcribe when: enough audio AND pause detected, or buffer too long
                should_flush = (len(buf) >= min_chunk_bytes and silence_run >= silence_trigger) \
                            or len(buf) >= max_chunk_bytes
                if should_flush:
                    chunk_dur = len(buf) / bps
                    _LOGGER.debug("transcribing %.1fs at offset=%.1fs (silence=%dms)",
                                  chunk_dur, time_offset, silence_run * 1000 // bps)
                    for line in self._transcribe_chunk(model, buf, time_offset):
                        _LOGGER.info("result: %s", line.decode().strip())
                        yield line
                    time_offset += chunk_dur
                    buf = b""
                    silence_run = 0

            # Flush remaining
            if buf and not self.cancelled:
                _LOGGER.info("flushing remaining %d bytes at offset=%.1fs", len(buf), time_offset)
                for line in self._transcribe_chunk(model, buf, time_offset):
                    _LOGGER.info("result: %s", line.decode().strip())
                    yield line
        finally:
            stop_ingest.set()
            ingest_thread.join(timeout=1.0)

    def _transcribe_chunk(self, model, pcm_bytes: bytes, time_offset: float) -> Iterator[bytes]:
        import numpy as np
        int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if int16.size == 0:
            return

        # Strict pre-gate: if the chunk is almost entirely silence/noise,
        # skip Whisper completely. This is the cheapest and most reliable
        # way to avoid subtitle-style hallucinations during silence while
        # still allowing short utterances to pass.
        frame_samples = max(1, self.sample_rate * _VOICE_FRAME_MS // 1000)
        voiced_frames = 0
        total_frames = 0
        peak_rms = 0
        for off in range(0, int16.size, frame_samples):
            frame = int16[off:off + frame_samples]
            if frame.size == 0:
                continue
            total_frames += 1
            rms = int(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            if rms > peak_rms:
                peak_rms = rms
            if rms >= _VOICE_FRAME_RMS:
                voiced_frames += 1
        voiced_ms = voiced_frames * _VOICE_FRAME_MS
        voiced_ratio = (voiced_frames / total_frames) if total_frames else 0.0
        if voiced_ms < _MIN_VOICED_MS and voiced_ratio < _MIN_VOICED_RATIO:
            _LOGGER.debug(
                "Skipping near-silence chunk at %.1fs (voiced_ms=%d, voiced_ratio=%.3f, peak_rms=%d)",
                time_offset,
                voiced_ms,
                voiced_ratio,
                peak_rms,
            )
            return

        samples = int16.astype(np.float32) / 32768.0
        segments, _ = model.transcribe(samples, language=self.language, beam_size=5,
                                       no_speech_threshold=0.98,
                                       condition_on_previous_text=False,
                                       hallucination_silence_threshold=0.2,
                                       vad_filter=True,
                                       vad_parameters={"threshold": 0.75,
                                                       "min_speech_duration_ms": 120,
                                                       "min_silence_duration_ms": 700,
                                                       "speech_pad_ms": 80})
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            # Filter hallucinations: high no_speech_prob or very low confidence
            if seg.no_speech_prob > 0.45:
                _LOGGER.debug("Skipping (no_speech_prob=%.2f): %s", seg.no_speech_prob, text)
                continue
            if seg.avg_logprob < -0.8:
                _LOGGER.debug("Skipping (avg_logprob=%.2f): %s", seg.avg_logprob, text)
                continue
            obj = {"text": text,
                   "start": round(seg.start + time_offset, 3),
                   "end": round(seg.end + time_offset, 3)}
            _LOGGER.info("STT [p=%.2f, ns=%.2f]: %s", seg.avg_logprob, seg.no_speech_prob, text)
            yield (json.dumps(obj, ensure_ascii=False) + "\n").encode()
