"""Shared pytest fixtures for speech-pipeline tests.

All tests use ephemeral ports (OS-assigned) to avoid conflicts
with the production server running on standard ports.
"""
import socket
import struct
import pytest


def find_free_port() -> int:
    """Get an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def generate_sine_pcm(freq_hz: float = 440.0, duration_s: float = 1.0,
                      sample_rate: int = 8000, amplitude: int = 8000) -> bytes:
    """Generate a sine wave as s16le PCM bytes."""
    import math
    samples = int(sample_rate * duration_s)
    data = []
    for i in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        data.append(struct.pack("<h", max(-32768, min(32767, value))))
    return b"".join(data)


def audio_similarity(reference: bytes, recorded: bytes) -> tuple:
    """Compare two s16le PCM buffers via normalized cross-correlation.

    Returns (similarity, delay_samples).
    similarity: 0.0-1.0 (1.0 = identical)
    delay_samples: positive = recorded is delayed
    """
    import numpy as np
    ref = np.frombuffer(reference, dtype=np.int16).astype(np.float64)
    rec = np.frombuffer(recorded, dtype=np.int16).astype(np.float64)

    if len(ref) == 0 or len(rec) == 0:
        return 0.0, 0

    correlation = np.correlate(rec, ref, mode="full")
    peak_idx = int(np.argmax(np.abs(correlation)))
    delay_samples = peak_idx - len(ref) + 1

    norm = float(np.sqrt(np.sum(ref ** 2) * np.sum(rec ** 2)))
    similarity = float(abs(correlation[peak_idx]) / norm) if norm > 0 else 0.0

    return similarity, delay_samples
