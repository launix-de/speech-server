"""Audio chain integrity tests — real audio through real Stage pipelines.

Loads examples/queue.mp3 (hold music), pushes it through the actual
SIPSource → converter → mixer chain, and asserts that the output
waveform still resembles the input (similarity > threshold).

A format mismatch (e.g. declaring u8 when data is s16le) inserts a
wrong converter that destroys the waveform → similarity drops to ~0.
"""
from __future__ import annotations

import audioop
import os
import threading
import queue
import time

import numpy as np

from speech_pipeline.base import AudioFormat, Stage
from speech_pipeline.SIPSource import SIPSource
from speech_pipeline.SampleRateConverter import SampleRateConverter
from speech_pipeline.EncodingConverter import EncodingConverter
from speech_pipeline.QueueSink import QueueSink

QUEUE_MP3 = os.path.join(os.path.dirname(__file__), "..", "examples", "queue.mp3")
_DURATION_S = 1.0  # first 1 second — fast enough for CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pcm(sample_rate: int) -> bytes:
    """First _DURATION_S seconds from queue.mp3 as mono s16le PCM."""
    import av
    container = av.open(QUEUE_MP3)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    pcm = b""
    max_bytes = int(sample_rate * _DURATION_S) * 2
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            pcm += bytes(out.planes[0])
        if len(pcm) >= max_bytes:
            break
    container.close()
    return pcm[:max_bytes]


def _chop(pcm: bytes, frame_bytes: int) -> list[bytes]:
    return [pcm[i:i + frame_bytes]
            for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes)]


def _similarity(ref: bytes, test: bytes) -> float:
    """Peak cross-correlation similarity — robust against codec delay."""
    a = np.frombuffer(ref, dtype=np.int16).astype(np.float64)
    b = np.frombuffer(test, dtype=np.int16).astype(np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    corr = np.correlate(b, a, mode="full")
    norm = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
    if norm == 0:
        return 0.0
    return float(np.max(np.abs(corr)) / norm)


def _fake_pyvoip_session(frames: list[bytes]):
    """Fake pyVoIP-like session with pre-loaded rx_queue."""
    rx_q = queue.Queue()
    for f in frames:
        rx_q.put(f)

    session = type("S", (), {
        "call": type("C", (), {})(),
        "connected": threading.Event(),
        "hungup": threading.Event(),
        "rx_queue": rx_q,
    })()
    session.connected.set()

    # Signal hangup after queue empties (SIPSource polls with 0.5s timeout)
    def _drain_then_hangup():
        while not rx_q.empty():
            time.sleep(0.02)
        time.sleep(0.2)
        session.hungup.set()
    threading.Thread(target=_drain_then_hangup, daemon=True).start()
    return session


def _fake_rtp_session(codec, frames: list[bytes]):
    """Fake RTPSession-backed session with pre-loaded rx_queue."""
    from speech_pipeline.RTPSession import RTPSession, RTPCallSession
    rtp = RTPSession.__new__(RTPSession)
    rtp.codec = codec
    rtp.rx_queue = queue.Queue()
    for f in frames:
        rtp.rx_queue.put(f)

    session = RTPCallSession.__new__(RTPCallSession)
    session._rtp = rtp
    session.connected = threading.Event()
    session.connected.set()
    session.hungup = threading.Event()
    session.rx_queue = rtp.rx_queue

    def _drain_then_hangup():
        while not rtp.rx_queue.empty():
            time.sleep(0.02)
        time.sleep(0.2)
        session.hungup.set()
    threading.Thread(target=_drain_then_hangup, daemon=True).start()
    return session


def _collect(source: Stage, sink_fmt: AudioFormat, timeout: float = 10) -> bytes:
    """Pipe source → QueueSink, drive pipeline, return collected PCM."""
    out_q = queue.Queue()
    sink = QueueSink(out_q, sink_fmt.sample_rate, sink_fmt.encoding)
    source.pipe(sink)
    t = threading.Thread(target=sink.run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    result = b""
    while True:
        try:
            chunk = out_q.get_nowait()
        except queue.Empty:
            break
        if chunk is None:
            break
        result += chunk
    return result


# ---------------------------------------------------------------------------
# pyVoIP path — this is where the u8 bug lived
# ---------------------------------------------------------------------------

class TestPyVoIPChain:
    """pyVoIP: SIPSource@8k → auto-convert → mixer@48k → back to 8k."""

    def test_pyvoip_to_mixer_similarity(self):
        """queue.mp3 through pyVoIP SIPSource → upsample → must match."""
        ref_8k = _load_pcm(8000)
        frames = _chop(ref_8k, 320)

        src = SIPSource(_fake_pyvoip_session(frames))
        out_48k = _collect(src, AudioFormat(48000, "s16le"))
        assert len(out_48k) > 0, "No output"

        down_8k, _ = audioop.ratecv(out_48k, 2, 1, 48000, 8000, None)
        sim = _similarity(ref_8k, down_8k)
        assert sim > 0.85, (
            f"pyVoIP→mixer similarity {sim:.3f} — audio distorted"
        )

    def test_pyvoip_round_trip_similarity(self):
        """queue.mp3: pyVoIP 8k → SRC↑48k → SRC↓8k, similarity.

        pipe() auto-inserts converters based on declared formats.
        If SIPSource wrongly declares u8, an EncodingConverter is
        inserted that destroys the audio → similarity drops to ~0.
        """
        ref_8k = _load_pcm(8000)
        frames = _chop(ref_8k, 320)

        src = SIPSource(_fake_pyvoip_session(frames))
        up = SampleRateConverter(8000, 48000)
        down = SampleRateConverter(48000, 8000)
        # pipe() checks SIPSource.output_format vs SRC.input_format.
        # If SIPSource declares u8 → pipe inserts EncodingConverter → boom.
        src.pipe(up)
        up.pipe(down)

        out_8k = _collect(down, AudioFormat(8000, "s16le"))
        assert len(out_8k) > 0, "No output"

        sim = _similarity(ref_8k, out_8k)
        assert sim > 0.80, (
            f"pyVoIP round-trip similarity {sim:.3f} — audio destroyed "
            f"(SIPSource format mismatch?)"
        )


# ---------------------------------------------------------------------------
# RTPSession path (PCMU 8kHz, Opus 48kHz)
# ---------------------------------------------------------------------------

class TestRTPChain:
    def test_pcmu_to_mixer_similarity(self):
        """PCMU@8k → upsample 48k, similarity with original."""
        from speech_pipeline.rtp_codec import PCMU
        ref_8k = _load_pcm(8000)
        frames = _chop(ref_8k, 320)

        src = SIPSource(_fake_rtp_session(PCMU, frames))
        out_48k = _collect(src, AudioFormat(48000, "s16le"))
        assert len(out_48k) > 0

        down_8k, _ = audioop.ratecv(out_48k, 2, 1, 48000, 8000, None)
        sim = _similarity(ref_8k, down_8k)
        assert sim > 0.85, f"PCMU→mixer similarity {sim:.3f}"

    def test_pcmu_full_round_trip_similarity(self):
        """PCMU: 8k → SRC↑48k → SRC↓8k, similarity with original."""
        from speech_pipeline.rtp_codec import PCMU
        ref_8k = _load_pcm(8000)
        frames = _chop(ref_8k, 320)

        src = SIPSource(_fake_rtp_session(PCMU, frames))
        up = SampleRateConverter(8000, 48000)
        down = SampleRateConverter(48000, 8000)
        src.pipe(up)
        up.pipe(down)

        out_8k = _collect(down, AudioFormat(8000, "s16le"))
        assert len(out_8k) > 0

        sim = _similarity(ref_8k, out_8k)
        assert sim > 0.85, f"PCMU round-trip similarity {sim:.3f}"

    def test_opus_passthrough_similarity(self):
        """Opus@48k → mixer@48k — no resampling, near-perfect similarity."""
        from speech_pipeline.rtp_codec import Opus
        ref_48k = _load_pcm(48000)
        frames = _chop(ref_48k, 1920)

        src = SIPSource(_fake_rtp_session(Opus.new_session_codec(), frames))
        out_48k = _collect(src, AudioFormat(48000, "s16le"))
        assert len(out_48k) > 0

        sim = _similarity(ref_48k, out_48k)
        assert sim > 0.95, (
            f"Opus 48k passthrough similarity {sim:.3f} — should be ~1.0"
        )


# ---------------------------------------------------------------------------
# Codec encode→decode round-trip with real audio
# ---------------------------------------------------------------------------

class TestCodecRoundTrip:
    def test_pcmu(self):
        from speech_pipeline.rtp_codec import PCMU
        ref = _load_pcm(8000)
        decoded = b"".join(PCMU.decode(PCMU.encode(f)) for f in _chop(ref, 320))
        sim = _similarity(ref, decoded)
        assert sim > 0.95, f"PCMU codec similarity {sim:.3f}"

    def test_pcma(self):
        from speech_pipeline.rtp_codec import PCMA
        ref = _load_pcm(8000)
        decoded = b"".join(PCMA.decode(PCMA.encode(f)) for f in _chop(ref, 320))
        sim = _similarity(ref, decoded)
        assert sim > 0.95, f"PCMA codec similarity {sim:.3f}"

    def test_opus(self):
        from speech_pipeline.rtp_codec import Opus
        codec = Opus.new_session_codec()
        try:
            ref = _load_pcm(48000)
            # Opus needs a few frames of warmup — encode all, skip first 5
            frames = _chop(ref, 1920)
            decoded_frames = []
            for f in frames:
                wire = codec.encode(f)
                decoded_frames.append(codec.decode(wire))
            # Skip first 5 frames (codec warmup), compare the rest
            warmup = 5
            ref_tail = b"".join(frames[warmup:])
            dec_tail = b"".join(decoded_frames[warmup:])
            sim = _similarity(ref_tail, dec_tail)
            assert sim > 0.80, f"Opus codec similarity {sim:.3f}"
        finally:
            codec.close()


# ---------------------------------------------------------------------------
# Regression proof: u8 misdeclaration destroys real audio
# ---------------------------------------------------------------------------

class TestEncodingCorruption:
    def test_u8_converter_destroys_s16le_audio(self):
        """Applying u8→s16le to already-s16le data must destroy audio."""
        ref = _load_pcm(8000)
        corrupted = EncodingConverter._u8_to_s16le(ref)
        assert len(corrupted) == len(ref) * 2  # each byte → 2 bytes
        min_len = min(len(ref), len(corrupted))
        sim = _similarity(ref[:min_len], corrupted[:min_len])
        assert sim < 0.3, (
            f"u8→s16le on s16le should destroy audio, sim={sim:.3f}"
        )
