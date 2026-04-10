"""Tests for PyVoIPCallSession — the adapter wrapping pyVoIP calls
so that SIPSource/SIPSink can treat them like RTPCallSession."""
import audioop
import queue
import struct
import threading
import time
from unittest.mock import MagicMock

import numpy as np

from speech_pipeline.telephony.leg import PyVoIPCallSession


def _make_session():
    mock_call = MagicMock()
    mock_call.read_audio.side_effect = Exception("stopped")
    return PyVoIPCallSession(mock_call), mock_call


class TestPyVoIPCallSessionInterface:
    """PyVoIPCallSession must expose connected, hungup, rx_queue, call —
    the same interface SIPSource/SIPSink rely on."""

    def test_connected_event_set_at_creation(self):
        session, _ = _make_session()
        assert isinstance(session.connected, threading.Event)
        assert session.connected.is_set()

    def test_hungup_event_unset_at_creation(self):
        session, _ = _make_session()
        assert isinstance(session.hungup, threading.Event)
        assert not session.hungup.is_set()

    def test_rx_queue_exists(self):
        session, _ = _make_session()
        assert isinstance(session.rx_queue, queue.Queue)

    def test_call_property_returns_underlying(self):
        session, mock_call = _make_session()
        assert session.call is mock_call


class TestPyVoIPCallSessionHangup:
    """hangup() must stop the rx pump, signal hungup, and forward to pyVoIP."""

    def test_hangup_sets_hungup_event(self):
        session, _ = _make_session()
        session.hangup()
        assert session.hungup.is_set()

    def test_hangup_stops_rx_pump(self):
        session, _ = _make_session()
        assert session._rx_pump_running is True
        session.hangup()
        assert session._rx_pump_running is False

    def test_hangup_calls_underlying_call(self):
        session, mock_call = _make_session()
        session.hangup()
        mock_call.hangup.assert_called_once()

    def test_hangup_tolerates_underlying_exception(self):
        session, mock_call = _make_session()
        mock_call.hangup.side_effect = RuntimeError("already dead")
        session.hangup()  # must not raise
        assert session.hungup.is_set()


# ---------------------------------------------------------------------------
# Codec detection: rx_pump must decode with the correct codec
# ---------------------------------------------------------------------------

def _load_pcm_frame(rate=8000):
    """Load one 20ms frame from examples/queue.mp3 as s16le PCM."""
    import os, av
    mp3 = os.path.join(os.path.dirname(__file__), "..", "examples", "queue.mp3")
    container = av.open(mp3)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    pcm = b""
    frame_bytes = int(rate * 0.02) * 2  # 20ms
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            pcm += bytes(out.planes[0])
        if len(pcm) >= frame_bytes:
            break
    container.close()
    return pcm[:frame_bytes]


def _make_session_with_codec(codec_name):
    """Create a PyVoIPCallSession with a mock pyVoIP call that uses a specific codec.

    Feeds 5 frames of encoded real audio, then raises to stop.
    """
    pcm = _load_pcm_frame(8000)  # 160 samples = 20ms from queue.mp3

    if codec_name == "alaw":
        encoded = audioop.lin2alaw(pcm, 2)
    else:
        encoded = audioop.lin2ulaw(pcm, 2)

    frame_count = [0]

    def read_audio(length=160, blocking=False):
        frame_count[0] += 1
        if frame_count[0] > 5:
            raise Exception("done")
        return encoded

    mock_call = MagicMock()
    mock_call.read_audio = read_audio

    # Set up RTPClients with codec preference
    mock_rtp = MagicMock()
    try:
        from pyVoIP.RTP import PayloadType
        if codec_name == "alaw":
            mock_rtp.preference = PayloadType.PCMA
        else:
            mock_rtp.preference = PayloadType.PCMU
    except ImportError:
        pass
    mock_call.RTPClients = [mock_rtp]

    session = PyVoIPCallSession(mock_call)
    return session, pcm


class TestPyVoIPCodecDetection:
    """rx_pump must use the correct decoder based on the negotiated codec."""

    def test_ulaw_decoded_correctly(self):
        """µ-law encoded audio must produce recognizable s16le output."""
        session, ref_pcm = _make_session_with_codec("ulaw")
        time.sleep(0.3)  # let rx_pump run

        collected = b""
        while not session.rx_queue.empty():
            collected += session.rx_queue.get_nowait()

        assert len(collected) > 0, "rx_pump produced no output"
        # Compare with reference — should be very similar
        n = min(len(ref_pcm), len(collected))
        a = np.frombuffer(ref_pcm[:n], dtype=np.int16).astype(np.float64)
        b = np.frombuffer(collected[:n], dtype=np.int16).astype(np.float64)
        corr = np.correlate(b, a, mode="full")
        norm = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
        sim = float(np.max(np.abs(corr)) / norm) if norm > 0 else 0
        assert sim > 0.9, f"µ-law decode similarity {sim:.3f} — wrong decoder?"
        session.hangup()

    def test_alaw_decoded_correctly(self):
        """A-law encoded audio must produce recognizable s16le output."""
        session, ref_pcm = _make_session_with_codec("alaw")
        time.sleep(0.3)

        collected = b""
        while not session.rx_queue.empty():
            collected += session.rx_queue.get_nowait()

        assert len(collected) > 0, "rx_pump produced no output"
        n = min(len(ref_pcm), len(collected))
        a = np.frombuffer(ref_pcm[:n], dtype=np.int16).astype(np.float64)
        b = np.frombuffer(collected[:n], dtype=np.int16).astype(np.float64)
        corr = np.correlate(b, a, mode="full")
        norm = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
        sim = float(np.max(np.abs(corr)) / norm) if norm > 0 else 0
        assert sim > 0.9, f"A-law decode similarity {sim:.3f} — wrong decoder?"
        session.hangup()

    def test_alaw_with_ulaw_decoder_distorts_audio(self):
        """Prove: decoding A-law with µ-law decoder distorts real audio."""
        pcm = _load_pcm_frame(8000)
        alaw_data = audioop.lin2alaw(pcm, 2)

        correct_pcm = audioop.alaw2lin(alaw_data, 2)
        wrong_pcm = audioop.ulaw2lin(alaw_data, 2)

        # Correct decoder should match original well
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        b_correct = np.frombuffer(correct_pcm, dtype=np.int16).astype(np.float64)
        b_wrong = np.frombuffer(wrong_pcm, dtype=np.int16).astype(np.float64)
        n = min(len(a), len(b_correct))

        norm_c = float(np.sqrt(np.sum(a[:n] ** 2) * np.sum(b_correct[:n] ** 2)))
        sim_correct = float(np.abs(np.dot(a[:n], b_correct[:n])) / norm_c) if norm_c else 0

        norm_w = float(np.sqrt(np.sum(a[:n] ** 2) * np.sum(b_wrong[:n] ** 2)))
        sim_wrong = float(np.abs(np.dot(a[:n], b_wrong[:n])) / norm_w) if norm_w else 0

        assert sim_correct > sim_wrong, (
            f"Wrong decoder ({sim_wrong:.3f}) should be worse than "
            f"correct decoder ({sim_correct:.3f})"
        )
