"""End-to-end conference audio quality test.

Simulates: inbound SIP call → ConferenceMixer → outbound SIP call.
Both sides send real audio (examples/queue.mp3); the other side must
receive it without distortion.

Tests every codec combination (PCMU, PCMA, G722).
"""
import itertools
import queue
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import find_free_port, audio_similarity
from speech_pipeline.ConferenceMixer import ConferenceMixer
from speech_pipeline.ConferenceLeg import ConferenceLeg
from speech_pipeline.SIPSource import SIPSource
from speech_pipeline.SIPSink import SIPSink
from speech_pipeline.RTPSession import RTPSession, RTPCallSession
from speech_pipeline.rtp_codec import PCMU, PCMA, G722, codec_for_pt

QUEUE_MP3 = Path(__file__).parent.parent / "examples" / "queue.mp3"
RTP_HEADER_SIZE = 12
DURATION_S = 0.8  # enough for quality check, short enough for fast tests


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode_mp3_to_s16le(sample_rate: int, duration_s: float = DURATION_S) -> bytes:
    """Decode queue.mp3 to s16le mono PCM at given sample rate."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(QUEUE_MP3), "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-t", str(duration_s), "-"],
        capture_output=True,
    )
    assert len(result.stdout) > 0, f"ffmpeg decode failed: {result.stderr.decode()}"
    return result.stdout


def _rtp_packet(payload: bytes, seq: int, ts: int, ssrc: int, pt: int) -> bytes:
    header = struct.pack("!BBHII", 0x80, pt, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
    return header + payload


def _send_pcm_as_rtp(sock, remote_addr, codec, pcm_s16le):
    """Segment s16le PCM into codec frames, encode, and send as RTP."""
    frame_bytes = codec.frame_samples * 2
    session_codec = codec.new_session_codec()
    seq, ts, ssrc = 0, 0, 0xABCD0001

    for i in range(0, len(pcm_s16le), frame_bytes):
        chunk = pcm_s16le[i:i + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk += b"\x00" * (frame_bytes - len(chunk))
        wire = session_codec.encode(chunk)
        pkt = _rtp_packet(wire, seq, ts, ssrc, codec.payload_type)
        sock.sendto(pkt, remote_addr)
        seq += 1
        ts += codec.timestamp_step
        time.sleep(0.018)

    session_codec.close()


def _collect_rtp_audio(sock, codec, expected_frames, timeout_s):
    """Receive RTP, decode, return s16le PCM."""
    import socket as _socket
    session_codec = codec.new_session_codec()
    frames = []
    deadline = time.monotonic() + timeout_s

    while len(frames) < expected_frames and time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(2048)
        except _socket.timeout:
            continue
        if len(data) < RTP_HEADER_SIZE:
            continue
        pt = data[1] & 0x7F
        if pt != codec.payload_type:
            continue
        pcm = session_codec.decode(data[RTP_HEADER_SIZE:])
        if pcm:
            frames.append(pcm)

    session_codec.close()
    return b"".join(frames)


def _rms(pcm_s16le: bytes) -> float:
    """Compute RMS power of s16le PCM."""
    if len(pcm_s16le) < 4:
        return 0.0
    samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(samples ** 2)))


# ---------------------------------------------------------------------------
# Conference E2E runner
# ---------------------------------------------------------------------------

def _run_conference_e2e(codec_a_name: str, codec_b_name: str):
    """Wire two RTP endpoints through a ConferenceMixer.

    A sends queue.mp3 audio, B receives it (and vice versa).
    Returns (audio_received_by_a, audio_received_by_b, codec_a, codec_b).
    """
    codec_a = codec_for_pt({"PCMU": 0, "PCMA": 8, "G722": 9}[codec_a_name])
    codec_b = codec_for_pt({"PCMU": 0, "PCMA": 8, "G722": 9}[codec_b_name])

    # Decode queue.mp3 at each codec's native sample rate
    pcm_a = _decode_mp3_to_s16le(codec_a.sample_rate)
    pcm_b = _decode_mp3_to_s16le(codec_b.sample_rate)

    # Ports
    port_a_local = find_free_port()
    port_a_remote = find_free_port()
    port_b_local = find_free_port()
    port_b_remote = find_free_port()

    # 1. Conference mixer
    mixer = ConferenceMixer("e2e", sample_rate=48000, frame_ms=20)
    mixer_thread = threading.Thread(target=mixer.run, daemon=True)
    mixer_thread.start()

    # 2. Leg A: RTP → SIPSource → ConferenceLeg → SIPSink → RTP
    rtp_a = RTPSession(port_a_local, "127.0.0.1", port_a_remote, codec=codec_a)
    rtp_a.start()
    session_a = RTPCallSession(rtp_a)
    src_a = SIPSource(session_a)
    leg_a = ConferenceLeg(sample_rate=48000)
    sink_a = SIPSink(session_a)
    src_a.pipe(leg_a)
    leg_a.attach(mixer)
    leg_a.pipe(sink_a)

    # SIPSink.run() drives the whole chain:
    #   sink → [SRC 48→codec] → ConferenceLeg → [SRC codec→48] → SIPSource
    # No separate drain thread needed — that would double-consume the generator.
    sink_a_thread = threading.Thread(target=sink_a.run, daemon=True)
    sink_a_thread.start()

    # 3. Leg B
    rtp_b = RTPSession(port_b_local, "127.0.0.1", port_b_remote, codec=codec_b)
    rtp_b.start()
    session_b = RTPCallSession(rtp_b)
    src_b = SIPSource(session_b)
    leg_b = ConferenceLeg(sample_rate=48000)
    sink_b = SIPSink(session_b)
    src_b.pipe(leg_b)
    leg_b.attach(mixer)
    leg_b.pipe(sink_b)

    sink_b_thread = threading.Thread(target=sink_b.run, daemon=True)
    sink_b_thread.start()

    time.sleep(0.15)  # let pipeline settle

    # 4. External sockets (simulating remote SIP endpoints)
    import socket
    sock_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_a.bind(("127.0.0.1", port_a_remote))
    sock_a.settimeout(0.1)

    sock_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_b.bind(("127.0.0.1", port_b_remote))
    sock_b.settimeout(0.1)

    expected_frames = int(DURATION_S / 0.02)

    # Start collection before sending
    received_a_list = []
    received_b_list = []

    def _collect_a():
        received_a_list.append(
            _collect_rtp_audio(sock_a, codec_a, expected_frames, DURATION_S + 3))

    def _collect_b():
        received_b_list.append(
            _collect_rtp_audio(sock_b, codec_b, expected_frames, DURATION_S + 3))

    t_collect_a = threading.Thread(target=_collect_a, daemon=True)
    t_collect_b = threading.Thread(target=_collect_b, daemon=True)
    t_collect_a.start()
    t_collect_b.start()

    # Send audio
    t_send_a = threading.Thread(
        target=_send_pcm_as_rtp,
        args=(sock_a, ("127.0.0.1", port_a_local), codec_a, pcm_a),
        daemon=True,
    )
    t_send_b = threading.Thread(
        target=_send_pcm_as_rtp,
        args=(sock_b, ("127.0.0.1", port_b_local), codec_b, pcm_b),
        daemon=True,
    )
    t_send_a.start()
    t_send_b.start()

    t_send_a.join(timeout=DURATION_S + 3)
    t_send_b.join(timeout=DURATION_S + 3)
    t_collect_a.join(timeout=DURATION_S + 4)
    t_collect_b.join(timeout=DURATION_S + 4)

    # Cleanup
    leg_a.cancel()
    leg_b.cancel()
    src_a.cancel()
    src_b.cancel()
    sink_a.cancel()
    sink_b.cancel()
    session_a.hungup.set()
    session_b.hungup.set()
    mixer.cancel()
    rtp_a.stop()
    rtp_b.stop()
    sock_a.close()
    sock_b.close()

    audio_a = received_a_list[0] if received_a_list else b""
    audio_b = received_b_list[0] if received_b_list else b""

    return audio_a, audio_b, codec_a, codec_b, pcm_a, pcm_b


# ---------------------------------------------------------------------------
# Tests: same codec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codec_name", ["PCMU", "PCMA", "G722"])
def test_same_codec(codec_name):
    """Both legs use the same codec. Received audio must resemble the
    original queue.mp3 signal — not silence, not noise."""
    audio_a, audio_b, codec_a, codec_b, pcm_a, pcm_b = \
        _run_conference_e2e(codec_name, codec_name)

    # Both sides must receive audio
    assert len(audio_a) > 0, f"Side A received nothing ({codec_name})"
    assert len(audio_b) > 0, f"Side B received nothing ({codec_name})"

    # Audio must not be silence
    rms_a = _rms(audio_a)
    rms_b = _rms(audio_b)
    assert rms_a > 100, \
        f"A←B [{codec_name}]: near-silent output (RMS={rms_a:.0f})"
    assert rms_b > 100, \
        f"B←A [{codec_name}]: near-silent output (RMS={rms_b:.0f})"

    # Correlation with original (A receives B's audio = pcm_b re-encoded)
    # Use shorter of received vs reference for comparison
    ref_b = _decode_mp3_to_s16le(codec_a.sample_rate)  # what A should hear
    ref_a = _decode_mp3_to_s16le(codec_b.sample_rate)  # what B should hear
    min_len = min(len(audio_a), len(ref_b), 8000)  # compare first ~0.5s at 8kHz
    if min_len > 2000:
        sim_a, _ = audio_similarity(ref_b[:min_len], audio_a[:min_len])
        assert sim_a > 0.3, \
            f"A←B [{codec_name}]: audio distorted (similarity={sim_a:.3f})"
    min_len = min(len(audio_b), len(ref_a), 8000)
    if min_len > 2000:
        sim_b, _ = audio_similarity(ref_a[:min_len], audio_b[:min_len])
        assert sim_b > 0.3, \
            f"B←A [{codec_name}]: audio distorted (similarity={sim_b:.3f})"


# ---------------------------------------------------------------------------
# Tests: cross-codec combinations
# ---------------------------------------------------------------------------

CODECS = ["PCMU", "PCMA", "G722"]


@pytest.mark.parametrize("codec_a,codec_b",
                         [(a, b) for a, b in itertools.product(CODECS, CODECS) if a != b])
def test_cross_codec(codec_a, codec_b):
    """Different codecs on each side. Audio must survive codec transcoding
    through the 48kHz conference mixer."""
    audio_a, audio_b, ca, cb, _, _ = _run_conference_e2e(codec_a, codec_b)

    assert len(audio_a) > 0, f"A received nothing ({codec_a}←{codec_b})"
    assert len(audio_b) > 0, f"B received nothing ({codec_a}→{codec_b})"

    rms_a = _rms(audio_a)
    rms_b = _rms(audio_b)
    assert rms_a > 100, \
        f"A←B [{codec_a}←{codec_b}]: near-silent (RMS={rms_a:.0f})"
    assert rms_b > 100, \
        f"B←A [{codec_a}→{codec_b}]: near-silent (RMS={rms_b:.0f})"

    # Similarity: A receives B's signal, B receives A's signal
    ref_for_a = _decode_mp3_to_s16le(ca.sample_rate)  # B sent this → A hears it
    ref_for_b = _decode_mp3_to_s16le(cb.sample_rate)  # A sent this → B hears it
    min_len = min(len(audio_a), len(ref_for_a), 8000)
    if min_len > 2000:
        sim_a, _ = audio_similarity(ref_for_a[:min_len], audio_a[:min_len])
        assert sim_a > 0.3, \
            f"A←B [{codec_a}←{codec_b}]: distorted (similarity={sim_a:.3f})"
    min_len = min(len(audio_b), len(ref_for_b), 8000)
    if min_len > 2000:
        sim_b, _ = audio_similarity(ref_for_b[:min_len], audio_b[:min_len])
        assert sim_b > 0.3, \
            f"B←A [{codec_a}→{codec_b}]: distorted (similarity={sim_b:.3f})"


# ---------------------------------------------------------------------------
# Test: bidirectional flow
# ---------------------------------------------------------------------------

def test_bidirectional_pcmu():
    """Sanity: both directions carry audio with PCMU."""
    audio_a, audio_b, _, _, _, _ = _run_conference_e2e("PCMU", "PCMU")
    assert len(audio_a) > 0, "A→B silent"
    assert len(audio_b) > 0, "B→A silent"
    assert _rms(audio_a) > 100, f"A←B near-silent (RMS={_rms(audio_a):.0f})"
    assert _rms(audio_b) > 100, f"B←A near-silent (RMS={_rms(audio_b):.0f})"
