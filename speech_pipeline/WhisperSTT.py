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


def _find_recent_pause_cut_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    target_bytes: int,
    search_window_bytes: int,
    rms_floor: int,
):
    """Return a smart split point near ``target_bytes`` if a recent pause exists.

    Searches the recent window before ``target_bytes`` for the longest silent
    run and returns its midpoint in bytes.  Returns ``None`` if there is no
    usable pause.
    """
    import numpy as np

    if target_bytes <= 0 or not pcm_bytes:
        return None

    bytes_per_sample = 2
    frame_samples = max(1, sample_rate * _VOICE_FRAME_MS // 1000)
    frame_bytes = frame_samples * bytes_per_sample
    if len(pcm_bytes) < frame_bytes:
        return None

    target_bytes = min(target_bytes, len(pcm_bytes))
    start_byte = max(0, target_bytes - search_window_bytes)
    start_byte -= start_byte % frame_bytes
    end_byte = target_bytes - (target_bytes % frame_bytes)
    if end_byte - start_byte < frame_bytes:
        return None

    samples = np.frombuffer(pcm_bytes[start_byte:end_byte], dtype=np.int16)
    best_run = None
    run_start = None
    total_frames = len(samples) // frame_samples
    for frame_idx in range(total_frames):
        off = frame_idx * frame_samples
        frame = samples[off:off + frame_samples]
        if frame.size == 0:
            continue
        rms = int(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        silent = rms < rms_floor
        if silent:
            if run_start is None:
                run_start = frame_idx
        elif run_start is not None:
            run_len = frame_idx - run_start
            if run_len > 0 and (best_run is None or run_len > best_run[2]):
                best_run = (run_start, frame_idx, run_len)
            run_start = None
    if run_start is not None:
        run_len = total_frames - run_start
        if run_len > 0 and (best_run is None or run_len > best_run[2]):
            best_run = (run_start, total_frames, run_len)

    if best_run is None:
        return None

    run_start_idx, run_end_idx, _ = best_run
    mid_frame = run_start_idx + ((run_end_idx - run_start_idx) // 2)
    cut = start_byte + (mid_frame * frame_bytes)
    cut -= cut % bytes_per_sample
    return cut if cut > 0 else None


def _choose_hard_split_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    max_chunk_bytes: int,
    search_window_bytes: int,
    rms_floor: int,
    overlap_bytes: int,
):
    """Choose a split point for forced chunking without a clean pause."""
    pause_cut = _find_recent_pause_cut_bytes(
        pcm_bytes,
        sample_rate=sample_rate,
        target_bytes=max_chunk_bytes,
        search_window_bytes=search_window_bytes,
        rms_floor=rms_floor,
    )
    if pause_cut is not None:
        return pause_cut

    split = max_chunk_bytes - overlap_bytes
    split = max(split, 2)
    split -= split % 2
    return split


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
        chunk_seconds = max(1.0, float(self.chunk_seconds))
        min_chunk_bytes = int(bps * chunk_seconds)
        max_chunk_bytes = int(bps * max(chunk_seconds * 5.0, 15.0))
        silence_trigger = int(bps * min(max(chunk_seconds * 0.25, 0.3), 0.8))
        overlap_bytes = int(bps * min(max(chunk_seconds * 0.10, 0.2), 0.3))
        rms_floor = _SILENCE_RMS_FLOOR        # int16 RMS below this = silence

        buf = b""
        time_offset = 0.0
        silence_run = 0

        _LOGGER.info(
            "WhisperTranscriber: pause-based chunking (silence=%dms, min=%.1fs, max=%.1fs, queue=%d)",
            silence_trigger * 1000 // bps,
            min_chunk_bytes / bps,
            max_chunk_bytes / bps,
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
                flush_on_pause = len(buf) >= min_chunk_bytes and silence_run >= silence_trigger
                flush_on_limit = len(buf) >= max_chunk_bytes
                if flush_on_pause or flush_on_limit:
                    flush_bytes = len(buf)
                    if flush_on_limit and not flush_on_pause:
                        flush_bytes = _choose_hard_split_bytes(
                            buf,
                            sample_rate=self.sample_rate,
                            max_chunk_bytes=max_chunk_bytes,
                            search_window_bytes=silence_trigger,
                            rms_floor=rms_floor,
                            overlap_bytes=overlap_bytes,
                        )
                    chunk = buf[:flush_bytes]
                    buf = buf[flush_bytes:]
                    chunk_dur = len(chunk) / bps
                    _LOGGER.debug("transcribing %.1fs at offset=%.1fs (silence=%dms)",
                                  chunk_dur, time_offset, silence_run * 1000 // bps)
                    for line in self._transcribe_chunk(model, chunk, time_offset):
                        _LOGGER.info("result: %s", line.decode().strip())
                        yield line
                    time_offset += chunk_dur
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
