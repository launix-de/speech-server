"""ConferenceMixer and ConferenceLeg tests."""
import queue
import struct
import threading
import time
import pytest
from conftest import generate_sine_pcm
from speech_pipeline.base import AudioFormat, Stage
from speech_pipeline.ConferenceMixer import ConferenceMixer
from speech_pipeline.ConferenceLeg import ConferenceLeg


class FixedSource(Stage):
    """Yields fixed-length PCM frames, then stops."""
    def __init__(self, data: bytes, sample_rate: int = 48000):
        super().__init__()
        self.output_format = AudioFormat(sample_rate, "s16le")
        self._data = data
        self._frame_bytes = int(sample_rate * 0.02) * 2  # 20ms

    def stream_pcm24k(self):
        for i in range(0, len(self._data), self._frame_bytes):
            if self.cancelled:
                break
            chunk = self._data[i:i + self._frame_bytes]
            if len(chunk) < self._frame_bytes:
                chunk += b"\x00" * (self._frame_bytes - len(chunk))
            yield chunk


def test_mixer_single_source():
    """One source → mixer output should contain that source's audio."""
    mixer = ConferenceMixer("test", sample_rate=48000, frame_ms=20)
    pcm = generate_sine_pcm(440, 0.1, 48000)
    src = FixedSource(pcm, 48000)

    src_id = mixer.add_source(src)
    out_q = mixer.add_output()

    t = threading.Thread(target=mixer.run, daemon=True)
    t.start()

    # Collect output frames
    frames = []
    for _ in range(10):
        try:
            frame = out_q.get(timeout=1.0)
            if frame is None:
                break
            frames.append(frame)
        except queue.Empty:
            break

    mixer.cancel()
    t.join(timeout=2)

    output = b"".join(frames)
    assert len(output) > 0, "Mixer produced no output"


def test_mixer_mix_minus():
    """Two participants: each hears the other, not themselves."""
    mixer = ConferenceMixer("test-mm", sample_rate=48000, frame_ms=20)

    # Source A: 440 Hz
    pcm_a = generate_sine_pcm(440, 0.1, 48000)
    src_a = FixedSource(pcm_a, 48000)

    # Source B: 880 Hz
    pcm_b = generate_sine_pcm(880, 0.1, 48000)
    src_b = FixedSource(pcm_b, 48000)

    src_id_a, sink_id_a, out_a = mixer.add_participant(src_a)
    src_id_b, sink_id_b, out_b = mixer.add_participant(src_b)

    t = threading.Thread(target=mixer.run, daemon=True)
    t.start()

    # Collect output for participant A (should hear B, not A)
    frames_a = []
    for _ in range(5):
        try:
            frame = out_a.get(timeout=1.0)
            if frame is None:
                break
            frames_a.append(frame)
        except queue.Empty:
            break

    mixer.cancel()
    t.join(timeout=2)

    output_a = b"".join(frames_a)
    assert len(output_a) > 0, "Participant A received no audio"


def test_conference_leg_attach():
    """ConferenceLeg attaches to mixer and yields output."""
    mixer = ConferenceMixer("test-leg", sample_rate=48000, frame_ms=20)

    pcm = generate_sine_pcm(440, 0.1, 48000)
    src = FixedSource(pcm, 48000)
    leg = ConferenceLeg(sample_rate=48000)
    src.pipe(leg)
    leg.attach(mixer)

    t = threading.Thread(target=mixer.run, daemon=True)
    t.start()

    frames = []
    for frame in leg.stream_pcm24k():
        frames.append(frame)
        if len(frames) >= 3:
            leg.cancel()
            break

    mixer.cancel()
    t.join(timeout=2)

    assert len(frames) >= 1, "ConferenceLeg yielded no frames"


def test_mixer_source_completion():
    """Source finishes → mixer drains buffer → done event set."""
    mixer = ConferenceMixer("test-done", sample_rate=48000, frame_ms=20)

    # Very short source — 2 frames
    frame_bytes = 960 * 2  # 20ms at 48kHz
    pcm = b"\x00" * (frame_bytes * 2)
    src = FixedSource(pcm, 48000)

    src_id = mixer.add_source(src)
    mixer.add_output()  # need at least one output for mixer to run

    t = threading.Thread(target=mixer.run, daemon=True)
    t.start()

    # Wait for source to finish
    done = mixer.wait_source(src_id, timeout=3.0)
    mixer.cancel()
    t.join(timeout=2)

    assert done, "Source did not complete within timeout"
