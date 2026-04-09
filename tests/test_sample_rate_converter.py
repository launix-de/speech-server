"""Tests for SampleRateConverter accuracy.

Every SIP call goes through resampling (8k/16k ↔ 48k). This test
verifies round-trip quality using real audio (examples/queue.mp3).
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import audio_similarity
from speech_pipeline.SampleRateConverter import SampleRateConverter
from speech_pipeline.base import AudioFormat, Stage

QUEUE_MP3 = Path(__file__).parent.parent / "examples" / "queue.mp3"


class FixedSource(Stage):
    """Yield s16le PCM in 20ms frames."""
    def __init__(self, data: bytes, sample_rate: int):
        super().__init__()
        self.output_format = AudioFormat(sample_rate, "s16le")
        self._data = data
        self._frame_bytes = int(sample_rate * 0.02) * 2

    def stream_pcm24k(self):
        for i in range(0, len(self._data), self._frame_bytes):
            if self.cancelled:
                break
            chunk = self._data[i:i + self._frame_bytes]
            if len(chunk) < self._frame_bytes:
                chunk += b"\x00" * (self._frame_bytes - len(chunk))
            yield chunk


def _decode_mp3(sample_rate: int, duration_s: float = 1.0) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-i", str(QUEUE_MP3), "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-t", str(duration_s), "-"],
        capture_output=True,
    )
    return result.stdout


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Run PCM through SampleRateConverter and collect output."""
    source = FixedSource(pcm, src_rate)
    converter = SampleRateConverter(src_rate, dst_rate)
    source.pipe(converter)
    return b"".join(converter.stream_pcm24k())


def _rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0


# ---------------------------------------------------------------------------
# One-way resampling: verify output has correct length and signal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src_rate,dst_rate", [
    (8000, 48000),
    (16000, 48000),
    (48000, 8000),
    (48000, 16000),
])
def test_resample_output_length(src_rate, dst_rate):
    """Output duration must match input duration (±1 frame tolerance)."""
    duration = 0.5
    pcm_in = _decode_mp3(src_rate, duration)
    pcm_out = _resample(pcm_in, src_rate, dst_rate)

    in_samples = len(pcm_in) // 2
    out_samples = len(pcm_out) // 2
    expected_samples = int(in_samples * dst_rate / src_rate)
    tolerance = dst_rate // 50  # 1 frame

    assert abs(out_samples - expected_samples) < tolerance, \
        f"{src_rate}→{dst_rate}: expected ~{expected_samples} samples, got {out_samples}"


@pytest.mark.parametrize("src_rate,dst_rate", [
    (8000, 48000),
    (16000, 48000),
    (48000, 8000),
    (48000, 16000),
])
def test_resample_preserves_signal(src_rate, dst_rate):
    """Output must not be silence — RMS must be substantial."""
    pcm_in = _decode_mp3(src_rate, 0.5)
    pcm_out = _resample(pcm_in, src_rate, dst_rate)

    rms_in = _rms(pcm_in)
    rms_out = _rms(pcm_out)

    assert rms_out > rms_in * 0.5, \
        f"{src_rate}→{dst_rate}: signal lost (RMS in={rms_in:.0f}, out={rms_out:.0f})"


# ---------------------------------------------------------------------------
# Round-trip: verify audio quality after up+down conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("native_rate", [8000, 16000])
def test_roundtrip_quality(native_rate):
    """native → 48k → native must preserve audio quality (similarity > 0.8)."""
    pcm_original = _decode_mp3(native_rate, 0.5)

    # Up-sample
    pcm_48k = _resample(pcm_original, native_rate, 48000)
    # Down-sample
    pcm_back = _resample(pcm_48k, 48000, native_rate)

    # Compare
    min_len = min(len(pcm_original), len(pcm_back))
    assert min_len > 2000, "Not enough audio for comparison"

    sim, delay = audio_similarity(pcm_original[:min_len], pcm_back[:min_len])
    assert sim > 0.8, \
        f"Round-trip {native_rate}→48k→{native_rate} distorted: similarity={sim:.3f}"
    assert abs(delay) < native_rate // 10, \
        f"Round-trip introduced excessive delay: {delay} samples"


# ---------------------------------------------------------------------------
# Pipe auto-insertion: verify pipe() inserts converter when needed
# ---------------------------------------------------------------------------

def test_pipe_auto_inserts_converter():
    """pipe() between different sample rates must auto-insert SampleRateConverter."""
    from speech_pipeline.base import _build_converter_chain
    src_fmt = AudioFormat(8000, "s16le")
    dst_fmt = AudioFormat(48000, "s16le")
    chain = _build_converter_chain(src_fmt, dst_fmt)
    assert len(chain) >= 1
    assert any(isinstance(s, SampleRateConverter) for s in chain)


def test_pipe_no_converter_same_rate():
    """pipe() between same sample rates must not insert SampleRateConverter."""
    from speech_pipeline.base import _build_converter_chain
    fmt = AudioFormat(48000, "s16le")
    chain = _build_converter_chain(fmt, fmt)
    assert len(chain) == 0
