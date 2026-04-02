"""RTP session send/receive tests using ephemeral ports."""
import queue
import socket
import struct
import threading
import time
import pytest
from conftest import find_free_port, generate_sine_pcm, audio_similarity


def test_rtp_loopback_pcmu():
    """Send PCM through RTPSession, receive on the other end."""
    from speech_pipeline.RTPSession import RTPSession
    from speech_pipeline.rtp_codec import PCMU

    port_a = find_free_port()
    port_b = find_free_port()

    codec_a = PCMU.new_session_codec()
    codec_b = PCMU.new_session_codec()

    # A sends to B, B sends to A
    sess_a = RTPSession(port_a, "127.0.0.1", port_b, codec=codec_a)
    sess_b = RTPSession(port_b, "127.0.0.1", port_a, codec=codec_b)

    sess_a.start()
    sess_b.start()

    try:
        pcm = generate_sine_pcm(440, 0.2, 8000)

        # Send from A in frame-sized chunks
        frame_size = PCMU.frame_samples * 2
        for i in range(0, len(pcm), frame_size):
            sess_a.write_s16le(pcm[i:i + frame_size])

        # Give TX loop time to send
        time.sleep(0.5)

        # Collect received data on B
        received = b""
        while True:
            try:
                frame = sess_b.rx_queue.get_nowait()
                received += frame
            except queue.Empty:
                break

        assert len(received) > 0, "No audio received"

        # Check similarity
        min_len = min(len(pcm), len(received))
        sim, delay = audio_similarity(pcm[:min_len], received[:min_len])
        assert sim > 0.9, f"RTP loopback similarity {sim:.3f} < 0.9"
    finally:
        sess_a.stop()
        sess_b.stop()
        codec_a.close()
        codec_b.close()


def test_rtp_loopback_opus():
    """Opus codec through RTP loopback."""
    from speech_pipeline.RTPSession import RTPSession
    from speech_pipeline.rtp_codec import Opus

    port_a = find_free_port()
    port_b = find_free_port()

    codec_a = Opus.new_session_codec()
    codec_b = Opus.new_session_codec()

    sess_a = RTPSession(port_a, "127.0.0.1", port_b, codec=codec_a)
    sess_b = RTPSession(port_b, "127.0.0.1", port_a, codec=codec_b)

    sess_a.start()
    sess_b.start()

    try:
        pcm = generate_sine_pcm(440, 0.2, 48000)

        frame_size = Opus.frame_samples * 2  # 960 * 2 = 1920
        for i in range(0, len(pcm), frame_size):
            chunk = pcm[i:i + frame_size]
            if len(chunk) == frame_size:
                sess_a.write_s16le(chunk)

        time.sleep(0.5)

        received = b""
        while True:
            try:
                frame = sess_b.rx_queue.get_nowait()
                received += frame
            except queue.Empty:
                break

        assert len(received) > 0, "No Opus audio received"
    finally:
        sess_a.stop()
        sess_b.stop()
        codec_a.close()
        codec_b.close()


def test_rtp_payload_type_filter():
    """RTP receiver ignores packets with wrong payload type."""
    from speech_pipeline.RTPSession import RTPSession, RTP_HEADER_SIZE
    from speech_pipeline.rtp_codec import PCMU

    port_rx = find_free_port()
    codec = PCMU.new_session_codec()
    sess = RTPSession(port_rx, "127.0.0.1", 1, codec=codec)
    sess.start()

    try:
        # Send a packet with wrong PT (99 instead of 0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        header = struct.pack("!BBHII", 0x80, 99, 0, 0, 0)
        payload = b"\xFF" * 160
        sock.sendto(header + payload, ("127.0.0.1", port_rx))
        sock.close()

        time.sleep(0.2)
        assert sess.rx_queue.empty(), "Wrong PT packet should be dropped"
    finally:
        sess.stop()
        codec.close()
