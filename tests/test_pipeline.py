"""Stage pipe() chain and auto-conversion tests."""
import pytest
from conftest import generate_sine_pcm, audio_similarity
from speech_pipeline.base import AudioFormat, Stage
from speech_pipeline.SampleRateConverter import SampleRateConverter


class DummySource(Stage):
    """Yields fixed PCM data."""
    def __init__(self, data: bytes, sample_rate: int = 8000):
        super().__init__()
        self.output_format = AudioFormat(sample_rate, "s16le")
        self._data = data
        self._chunk_size = 320  # 20ms at 8kHz

    def stream_pcm24k(self):
        for i in range(0, len(self._data), self._chunk_size):
            if self.cancelled:
                break
            yield self._data[i:i + self._chunk_size]


class CollectorSink(Stage):
    """Collects all yielded data."""
    def __init__(self, sample_rate: int = 8000):
        super().__init__()
        self.input_format = AudioFormat(sample_rate, "s16le")
        self.collected = b""

    def stream_pcm24k(self):
        for chunk in self.upstream.stream_pcm24k():
            self.collected += chunk
            yield chunk


def test_direct_pipe_same_format():
    """Source → Sink at same sample rate, no converter needed."""
    pcm = generate_sine_pcm(440, 0.2, 8000)
    src = DummySource(pcm, 8000)
    sink = CollectorSink(8000)
    src.pipe(sink)
    list(sink.stream_pcm24k())  # drain
    assert sink.collected == pcm


def test_pipe_auto_resample():
    """Source at 8kHz → Sink at 48kHz inserts SampleRateConverter."""
    pcm_8k = generate_sine_pcm(440, 0.1, 8000)
    src = DummySource(pcm_8k, 8000)
    sink = CollectorSink(48000)
    src.pipe(sink)
    list(sink.stream_pcm24k())  # drain

    # Output should be 6x longer (48000/8000)
    expected_samples = len(pcm_8k) // 2 * 6
    actual_samples = len(sink.collected) // 2
    assert abs(actual_samples - expected_samples) < 50, \
        f"Expected ~{expected_samples} samples, got {actual_samples}"


def test_sample_rate_converter_round_trip():
    """8k → 48k → 8k should preserve audio shape."""
    pcm_8k = generate_sine_pcm(440, 0.2, 8000)
    src = DummySource(pcm_8k, 8000)
    up = SampleRateConverter(8000, 48000)
    down = SampleRateConverter(48000, 8000)
    src.pipe(up)
    up.pipe(down)

    result = b""
    for chunk in down.stream_pcm24k():
        result += chunk

    min_len = min(len(pcm_8k), len(result))
    sim, delay = audio_similarity(pcm_8k[:min_len], result[:min_len])
    assert sim > 0.9, f"Resample round-trip similarity {sim:.3f} < 0.9"


def test_pipe_chain_multiple_stages():
    """Source → Stage1 → Stage2 → Sink chains correctly."""
    pcm = generate_sine_pcm(440, 0.1, 16000)
    src = DummySource(pcm, 16000)
    src._chunk_size = 640  # 20ms at 16kHz
    mid = SampleRateConverter(16000, 48000)
    sink = CollectorSink(48000)

    src.pipe(mid)
    mid.pipe(sink)
    list(sink.stream_pcm24k())

    # 3x upsample
    expected_samples = len(pcm) // 2 * 3
    actual_samples = len(sink.collected) // 2
    assert abs(actual_samples - expected_samples) < 50


def test_cancel_propagates():
    """Cancelling a stage stops the chain."""
    pcm = generate_sine_pcm(440, 1.0, 8000)  # 1 second
    src = DummySource(pcm, 8000)
    sink = CollectorSink(8000)
    src.pipe(sink)

    count = 0
    for chunk in sink.stream_pcm24k():
        count += 1
        if count >= 3:
            src.cancel()
            break

    # Should have stopped after ~3 chunks, not processed the full 1s
    assert len(sink.collected) < len(pcm)
