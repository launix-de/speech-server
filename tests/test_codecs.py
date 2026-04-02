"""Codec encode/decode round-trip tests."""
import struct
import pytest
from conftest import generate_sine_pcm, audio_similarity


def _round_trip(codec_singleton, sample_rate):
    """Encode → decode round-trip for a codec."""
    codec = codec_singleton.new_session_codec()
    try:
        pcm = generate_sine_pcm(440, 0.5, sample_rate)

        # Process in frame-sized chunks
        frame_bytes = codec.frame_samples * 2  # s16le
        decoded = b""
        for i in range(0, len(pcm), frame_bytes):
            frame = pcm[i:i + frame_bytes]
            if len(frame) < frame_bytes:
                frame += b"\x00" * (frame_bytes - len(frame))
            wire = codec.encode(frame)
            assert wire, f"Encode returned empty for {codec.name}"
            dec = codec.decode(wire)
            assert len(dec) == frame_bytes, \
                f"{codec.name}: decode returned {len(dec)} bytes, expected {frame_bytes}"
            decoded += dec

        # Compare: codec is lossy, so allow some degradation
        min_len = min(len(pcm), len(decoded))
        sim, delay = audio_similarity(pcm[:min_len], decoded[:min_len])
        assert sim > 0.8, f"{codec.name} round-trip similarity {sim:.3f} < 0.8"
        # Codecs like G.722 and Opus have inherent encoder delay (up to ~40ms)
        max_delay = codec.frame_samples * 2  # allow up to two frames of delay
        assert abs(delay) < max_delay, \
            f"{codec.name} round-trip delay {delay} samples > {max_delay}"
    finally:
        codec.close()


def test_pcmu_round_trip():
    from speech_pipeline.rtp_codec import PCMU
    _round_trip(PCMU, 8000)


def test_pcma_round_trip():
    from speech_pipeline.rtp_codec import PCMA
    _round_trip(PCMA, 8000)


def test_g722_round_trip():
    from speech_pipeline.rtp_codec import G722
    _round_trip(G722, 16000)


def test_opus_round_trip():
    from speech_pipeline.rtp_codec import Opus
    _round_trip(Opus, 48000)


def test_codec_preference_order():
    from speech_pipeline.rtp_codec import CODEC_PREFERENCE, Opus, G722, PCMU, PCMA
    assert CODEC_PREFERENCE[0] is Opus
    assert CODEC_PREFERENCE[1] is G722
    assert CODEC_PREFERENCE[2] is PCMU
    assert CODEC_PREFERENCE[3] is PCMA


def test_negotiate_payload_type():
    from speech_pipeline.rtp_codec import negotiate_payload_type
    # Remote offers G722 + PCMU → we prefer Opus but it's not offered → G722
    assert negotiate_payload_type([9, 0]) == 9
    # Remote offers only PCMA → we accept
    assert negotiate_payload_type([8]) == 8
    # Remote offers Opus → we prefer it
    assert negotiate_payload_type([111, 9, 0, 8]) == 111
    # Remote offers nothing we know → fallback PCMU
    assert negotiate_payload_type([99, 100]) == 0


def test_codec_for_pt():
    from speech_pipeline.rtp_codec import codec_for_pt
    c = codec_for_pt(111)
    assert c is not None
    assert c.name == "opus"
    assert c.sample_rate == 48000
    c.close()

    c2 = codec_for_pt(9)
    assert c2 is not None
    assert c2.name == "G722"
    c2.close()

    assert codec_for_pt(99) is None


def test_sdp_rtpmap():
    from speech_pipeline.rtp_codec import Opus, G722, PCMU
    assert "opus/48000/2" in Opus.sdp_rtpmap
    assert "fmtp" in Opus.sdp_rtpmap  # Opus needs fmtp line
    assert "G722/8000" in G722.sdp_rtpmap
    assert "PCMU/8000" in PCMU.sdp_rtpmap
