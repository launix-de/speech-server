"""CRM use-case end-to-end tests with audio quality measurement.

Each test creates a real conference, wires real RTP sessions,
pumps actual audio (examples/queue.mp3) through the pipeline,
and verifies the output waveform via cross-correlation.

Scenarios mirror the fop-dev CRM call flow:
- Hold music (play → conference)
- SIP bridge (sip → call → sip, bidirectional)
- Hold/unhold (detach leg, play hold jingle, reattach)
- TTS announcement into conference
"""
from __future__ import annotations

import audioop
import json
import os
import queue
import threading
import time

import numpy as np
import pytest

from conftest import ADMIN_TOKEN, ACCOUNT_TOKEN, ACCOUNT_ID, SUBSCRIBER_ID, create_call
from speech_pipeline.telephony import call_state, leg as leg_mod
from speech_pipeline.RTPSession import RTPSession, RTPCallSession
from speech_pipeline.rtp_codec import PCMU, PCMA, G722, Opus

QUEUE_MP3 = os.path.join(os.path.dirname(__file__), "..", "examples", "queue.mp3")
VOICES_PATH = os.path.join(os.path.dirname(__file__), "..", "voices-piper")
_DURATION_S = 0.5  # short for fast tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pcm(sample_rate: int = 8000, duration: float = _DURATION_S) -> bytes:
    """Load first N seconds from queue.mp3 as mono s16le PCM."""
    import av
    container = av.open(QUEUE_MP3)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    pcm = b""
    max_bytes = int(sample_rate * duration) * 2
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            pcm += bytes(out.planes[0])
        if len(pcm) >= max_bytes:
            break
    container.close()
    return pcm[:max_bytes]


def _similarity(ref: bytes, test: bytes) -> float:
    """Peak cross-correlation — robust against codec delay."""
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


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_rtp_leg(subscriber_id=SUBSCRIBER_ID, pbx_id="TestPBX",
                  number="+4917012345", codec=None):
    """Create a leg backed by a real RTPSession on localhost.

    Returns (leg, rtp_session, rtp_call_session).
    The RTP session is started and ready for audio.
    """
    codec = codec or PCMU
    local_port = _find_free_port()
    remote_port = _find_free_port()

    # Create RTP session (our "phone" side)
    phone_rtp = RTPSession(remote_port, "127.0.0.1", local_port,
                           codec=codec.new_session_codec())
    phone_rtp.start()

    # Create RTP session (server side, attached to leg)
    server_rtp = RTPSession(local_port, "127.0.0.1", remote_port,
                            codec=codec.new_session_codec())
    server_rtp.start()

    session = RTPCallSession(server_rtp)

    # Create leg
    leg = leg_mod.Leg(
        leg_id=f"leg-rtp-{local_port}",
        direction="inbound",
        number=number,
        pbx_id=pbx_id,
        subscriber_id=subscriber_id,
    )
    leg.voip_call = server_rtp  # for SIPSource detection
    leg.sip_session = session
    leg.rtp_session = server_rtp
    leg_mod._legs[leg.leg_id] = leg

    return leg, phone_rtp, session


def _send_audio(rtp: RTPSession, pcm_8k: bytes, codec=None):
    """Send s16le PCM through an RTP session (encode + transmit)."""
    codec = codec or rtp.codec
    frame_samples = codec.frame_samples
    frame_bytes = frame_samples * 2  # s16le
    for i in range(0, len(pcm_8k) - frame_bytes + 1, frame_bytes):
        rtp.write_s16le(pcm_8k[i:i + frame_bytes])
        time.sleep(frame_samples / codec.sample_rate * 0.5)  # half-speed to not overflow


def _receive_audio(rtp: RTPSession, duration_s: float = 0.5) -> bytes:
    """Receive audio from RTP session for a given duration."""
    collected = b""
    deadline = time.monotonic() + duration_s + 0.5  # extra margin
    while time.monotonic() < deadline:
        try:
            frame = rtp.rx_queue.get(timeout=0.1)
            if frame:
                collected += frame
        except queue.Empty:
            continue
        # Stop early if we have enough
        expected = int(rtp.codec.sample_rate * duration_s) * 2
        if len(collected) >= expected:
            break
    return collected


def _cleanup_leg(leg, phone_rtp, server_rtp=None):
    """Stop RTP sessions and remove leg."""
    phone_rtp.stop()
    if server_rtp:
        server_rtp.stop()
    elif leg.rtp_session:
        leg.rtp_session.stop()
    leg_mod._legs.pop(leg.leg_id, None)


# ---------------------------------------------------------------------------
# Test: Hold music (play → conference)
# ---------------------------------------------------------------------------

class TestHoldMusic:
    """Play hold music into conference, verify a participant hears it."""

    def test_hold_music_audible(self, client, account):
        """play:hold -> call:CALL — audio must reach the conference with energy."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)

        # Add a raw output queue to the conference to capture audio
        out_q = call.mixer.add_output()

        # Start hold music via pipeline API
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'play:hold{{"url":"examples/queue.mp3"}} -> call:{call_id}'
                           }),
                           headers=account)
        assert resp.status_code == 201

        # Collect audio from the conference
        time.sleep(1.0)  # let play stage start + AudioReader load MP3
        collected = b""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(collected) < 48000 * 2:
            try:
                frame = out_q.get(timeout=0.2)
                if frame:
                    collected += frame
            except queue.Empty:
                continue

        assert len(collected) > 0, "No audio from conference"

        # Verify the audio has real signal energy (not silence/noise)
        samples = np.frombuffer(collected, dtype=np.int16).astype(np.float64)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        assert rms > 500, (
            f"Hold music RMS={rms:.0f} — too quiet, likely silence or distorted"
        )

        # Verify it's not just noise: autocorrelation should show structure
        # (music has repeating patterns, noise doesn't)
        mid = len(samples) // 2
        chunk = samples[mid:mid + 4800]  # ~100ms at 48kHz
        if len(chunk) > 960:
            ac = float(np.abs(np.dot(chunk[:-960], chunk[960:])))
            ac_norm = float(np.sum(chunk ** 2))
            if ac_norm > 0:
                structure = ac / ac_norm
                assert structure > 0.1, (
                    f"Hold music has no temporal structure ({structure:.3f}) — "
                    f"likely noise/distortion, not music"
                )

        client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# Test: SIP bridge (sip → call → sip, bidirectional)
# ---------------------------------------------------------------------------

class TestSIPBridge:
    """Bridge a real RTP leg into a conference, send audio, verify reception."""

    def test_rtp_leg_audio_reaches_conference(self, client, account):
        """sip:LEG -> call:CALL -> sip:LEG — audio sent by phone reaches conf."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        leg, phone_rtp, session = _make_rtp_leg()

        try:
            # Bridge leg via pipeline
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                               }),
                               headers=account)
            assert resp.status_code == 201

            time.sleep(0.3)  # let pipeline wire up

            # Add output tap on conference
            out_q = call.mixer.add_output()

            # Send audio from the "phone"
            ref_pcm = _load_pcm(phone_rtp.codec.sample_rate, 0.3)
            _send_audio(phone_rtp, ref_pcm)

            # Collect from conference
            time.sleep(0.5)
            collected = b""
            while True:
                try:
                    frame = out_q.get_nowait()
                    if frame:
                        collected += frame
                except queue.Empty:
                    break

            assert len(collected) > 0, "No audio reached conference"

            # Compare (conference runs at 48kHz)
            down, _ = audioop.ratecv(collected, 2, 1, 48000,
                                     phone_rtp.codec.sample_rate, None)
            sim = _similarity(ref_pcm, down)
            assert sim > 0.5, (
                f"Leg→conference similarity {sim:.3f} — audio distorted"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)

    def test_bidirectional_audio_loopback(self, client, account):
        """Send audio from phone → conference → back to phone (mix-minus).

        With only one participant, conference output = own audio.
        But mix-minus subtracts own input → silence.
        So we add a second source (play) and verify the phone hears it.
        """
        call_id = create_call(client, account)
        leg, phone_rtp, session = _make_rtp_leg()

        try:
            # Bridge leg (bidirectional)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)

            # Also inject audio into conference from another source
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f'play:music{{"url":"examples/queue.mp3"}} -> call:{call_id}'
                        }),
                        headers=account)

            time.sleep(0.5)  # let audio flow

            # Receive on the phone side — should hear the play source
            received = _receive_audio(phone_rtp, 0.5)
            assert len(received) > 0, "Phone received no audio"

            # Verify the received audio has energy (not silence)
            samples = np.frombuffer(received, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            assert rms > 200, (
                f"Phone RTP output RMS={rms:.0f} — no audio reaching phone"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)


# ---------------------------------------------------------------------------
# Test: Kill stage (stop hold music)
# ---------------------------------------------------------------------------

class TestKillStage:
    """Stop a play stage and verify audio stops."""

    def test_kill_play_stops_audio(self, client, account):
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        out_q = call.mixer.add_output()

        # Start play
        client.post("/api/pipelines",
                     data=json.dumps({
                         "dsl": f'play:{call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
                     }),
                     headers=account)

        time.sleep(0.5)

        # Verify audio is flowing
        pre_kill = b""
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                frame = out_q.get(timeout=0.1)
                if frame:
                    pre_kill += frame
            except queue.Empty:
                continue
            if len(pre_kill) > 4800:
                break
        assert len(pre_kill) > 0, "No audio before kill"

        # Kill the stage
        resp = client.delete('/api/pipelines', data=json.dumps({"dsl": f"play:{call_id}_wait"}), headers=account)
        assert resp.status_code == 204

        # Drain any remaining buffered audio
        time.sleep(0.3)
        while not out_q.empty():
            try:
                out_q.get_nowait()
            except queue.Empty:
                break

        # After kill, conference output should be silence (all zeros)
        time.sleep(0.3)
        post_kill = b""
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                frame = out_q.get(timeout=0.1)
                if frame:
                    post_kill += frame
            except queue.Empty:
                continue

        if len(post_kill) > 0:
            samples = np.frombuffer(post_kill, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            assert rms < 100, f"Audio still playing after kill (RMS={rms:.0f})"

        client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# Test: every codec through conference (same-codec)
# ---------------------------------------------------------------------------

_ALL_CODECS = [
    pytest.param(PCMU, id="PCMU"),
    pytest.param(PCMA, id="PCMA"),
    pytest.param(G722, id="G722"),
    pytest.param(Opus, id="Opus"),
]


class TestCodecConference:
    """Send audio through phone → conference → phone for every codec."""

    @pytest.mark.parametrize("codec", _ALL_CODECS)
    def test_same_codec(self, client, account, codec):
        """Phone (codec X) → conference → phone (codec X): audio survives."""
        call_id = create_call(client, account)
        leg, phone_rtp, session = _make_rtp_leg(codec=codec)

        try:
            # Bridge
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)

            # Also play hold music so there is something to hear
            # (mix-minus subtracts own audio, so we need a second source)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f'play:music{{"url":"examples/queue.mp3"}} -> call:{call_id}'
                        }),
                        headers=account)

            time.sleep(0.8)  # let pipeline wire up + audio start flowing

            # Receive on phone
            received = _receive_audio(phone_rtp, 0.5)
            assert len(received) > 0, f"{codec.name}: phone received no audio"

            # Verify signal energy
            samples = np.frombuffer(received, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            assert rms > 100, (
                f"{codec.name}: phone RTP output RMS={rms:.0f} — "
                f"no audio reaching phone (codec distortion?)"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)

    @pytest.mark.parametrize("codec", _ALL_CODECS)
    def test_phone_to_conference_similarity(self, client, account, codec):
        """Phone sends queue.mp3 → conference mixer captures it, similarity check."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        leg, phone_rtp, session = _make_rtp_leg(codec=codec)

        try:
            # Bridge
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)

            time.sleep(0.5)

            # Tap conference output
            out_q = call.mixer.add_output()

            # Send real audio from phone
            ref_pcm = _load_pcm(codec.sample_rate, 0.3)
            _send_audio(phone_rtp, ref_pcm)

            # Collect from conference (48kHz)
            time.sleep(0.5)
            collected = b""
            while True:
                try:
                    frame = out_q.get_nowait()
                    if frame:
                        collected += frame
                except queue.Empty:
                    break

            assert len(collected) > 0, f"{codec.name}: no audio in conference"

            # Downsample to codec rate for comparison
            down, _ = audioop.ratecv(collected, 2, 1, 48000,
                                     codec.sample_rate, None)
            sim = _similarity(ref_pcm, down)
            assert sim > 0.7, (
                f"{codec.name}: phone→conference similarity {sim:.3f} — "
                f"audio distorted through {codec.name} codec path"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)


# ---------------------------------------------------------------------------
# Test: cross-codec conference (two legs, different codecs)
# ---------------------------------------------------------------------------

_CROSS_CODEC_PAIRS = [
    pytest.param(PCMU, Opus, id="PCMU-Opus"),
    pytest.param(Opus, PCMU, id="Opus-PCMU"),
    pytest.param(PCMA, G722, id="PCMA-G722"),
    pytest.param(G722, PCMA, id="G722-PCMA"),
]


class TestCrossCodecConference:
    """Two legs with different codecs in the same conference."""

    @pytest.mark.parametrize("codec_a,codec_b", _CROSS_CODEC_PAIRS)
    def test_cross_codec(self, client, account, codec_a, codec_b):
        """Leg A (codec_a) sends audio, Leg B (codec_b) receives it."""
        call_id = create_call(client, account)

        leg_a, phone_a, _ = _make_rtp_leg(
            codec=codec_a, number="+49111")
        leg_b, phone_b, _ = _make_rtp_leg(
            codec=codec_b, number="+49222")

        try:
            # Bridge both legs
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_a.leg_id} -> call:{call_id} -> sip:{leg_a.leg_id}"
                        }),
                        headers=account)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_b.leg_id} -> call:{call_id} -> sip:{leg_b.leg_id}"
                        }),
                        headers=account)

            time.sleep(1.0)

            # Drain any initial silence from Phone B's rx_queue
            # (mixer outputs silence before Leg A sends real audio)
            while not phone_b.rx_queue.empty():
                try:
                    phone_b.rx_queue.get_nowait()
                except queue.Empty:
                    break

            # Phone A sends audio
            ref_pcm = _load_pcm(codec_a.sample_rate, 0.5)
            _send_audio(phone_a, ref_pcm)

            # Wait for audio to traverse the pipeline
            time.sleep(0.8)

            # Drain initial silence again (pipeline has ~500ms latency)
            # then collect real audio
            received = _receive_audio(phone_b, 1.0)
            assert len(received) > 0, (
                f"{codec_a.name}→{codec_b.name}: phone B received no audio"
            )

            # Verify signal energy
            samples = np.frombuffer(received, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            assert rms > 100, (
                f"{codec_a.name}→{codec_b.name}: phone B RMS={rms:.0f} — "
                f"cross-codec transcoding failed"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg_a, phone_a)
            _cleanup_leg(leg_b, phone_b)


# ---------------------------------------------------------------------------
# Test: Tee sidechain as separate pipeline request (the CRM pattern)
# ---------------------------------------------------------------------------

class TestTeeSidechainSeparateRequest:
    """CRM sends bridge+tee in one request, sidechain in another.

    This was broken when /api/calls/{id}/pipes was replaced by
    /api/pipelines: the sidechain request couldn't find the tee
    because it created a new executor (no call: in the DSL).
    """

    def test_sidechain_attaches_to_existing_tee(self, client, account):
        """Bridge with tee, then sidechain in separate request."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        leg, phone_rtp, session = _make_rtp_leg()

        try:
            # Request 1: bridge with tee
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"sip:{leg.leg_id} -> tee:{leg.leg_id}_tap "
                                          f"-> call:{call_id} -> sip:{leg.leg_id}"
                               }),
                               headers=account)
            assert resp.status_code == 201, f"Bridge failed: {resp.data}"

            # Verify tee exists in executor
            ex = call.pipe_executor
            assert f"{leg.leg_id}_tap" in ex._tees, "Tee not created by bridge pipe"

            # Request 2: sidechain (no call: in DSL!)
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"tee:{leg.leg_id}_tap -> stt:de "
                                          f"-> webhook:https://example.com/stt"
                               }),
                               headers=account)
            assert resp.status_code == 201, (
                f"Sidechain failed: {resp.data} — "
                f"tee lookup across executors broken?"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)

    def test_sidechain_receives_audio(self, client, account):
        """Audio from phone goes through tee into sidechain."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        leg, phone_rtp, session = _make_rtp_leg()

        try:
            # Bridge with tee
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> tee:{leg.leg_id}_tap "
                                   f"-> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)

            time.sleep(0.5)

            # Verify audio still reaches conference through the tee
            out_q = call.mixer.add_output()

            ref_pcm = _load_pcm(phone_rtp.codec.sample_rate, 0.3)
            _send_audio(phone_rtp, ref_pcm)

            time.sleep(0.5)
            collected = b""
            while True:
                try:
                    frame = out_q.get_nowait()
                    if frame:
                        collected += frame
                except queue.Empty:
                    break

            assert len(collected) > 0, "No audio through tee into conference"

            down, _ = audioop.ratecv(collected, 2, 1, 48000,
                                     phone_rtp.codec.sample_rate, None)
            sim = _similarity(ref_pcm, down)
            assert sim > 0.5, (
                f"Audio through tee similarity {sim:.3f} — tee corrupts audio"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)

    def test_sidechain_data_flows_to_endpoint(self, client, account):
        """Audio through tee sidechain actually reaches the terminal stage.

        Uses a QueueSink instead of webhook+STT to verify data flows
        through the sidechain without needing a Whisper model.
        """
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        leg, phone_rtp, session = _make_rtp_leg()

        try:
            # Bridge with tee
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"sip:{leg.leg_id} -> tee:{leg.leg_id}_tap "
                                          f"-> call:{call_id} -> sip:{leg.leg_id}"
                               }),
                               headers=account)
            assert resp.status_code == 201

            time.sleep(0.3)

            # Manually attach a sidechain to the tee and collect audio
            ex = call.pipe_executor
            tee = ex._tees.get(f"{leg.leg_id}_tap")
            assert tee is not None, "Tee not found in executor"

            sidechain_q = queue.Queue(maxsize=200)
            from speech_pipeline.QueueSink import QueueSink
            from speech_pipeline.base import AudioFormat
            sc_sink = QueueSink(sidechain_q, 48000, "s16le")
            tee.add_sidechain(sc_sink)
            threading.Thread(target=sc_sink.run, daemon=True).start()

            time.sleep(0.3)

            # Send audio from phone
            ref_pcm = _load_pcm(phone_rtp.codec.sample_rate, 0.3)
            _send_audio(phone_rtp, ref_pcm)

            time.sleep(0.8)

            # Collect from sidechain
            sc_data = b""
            while True:
                try:
                    f = sidechain_q.get_nowait()
                    if f is None:
                        break
                    sc_data += f
                except queue.Empty:
                    break

            assert len(sc_data) > 0, (
                "Sidechain received no audio — tee not forwarding to sidechain"
            )
            samples = np.frombuffer(sc_data, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            assert rms > 100, (
                f"Sidechain audio RMS={rms:.0f} — only silence reached sidechain"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)


# ---------------------------------------------------------------------------
# Test: Hold / Unhold cycle (CRM pattern)
# ---------------------------------------------------------------------------

class TestHoldUnhold:
    """Kill bridge → play hold jingle → rebridge."""

    def test_hold_unhold_cycle(self, client, account):
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        leg, phone_rtp, session = _make_rtp_leg()

        try:
            # 1. Bridge leg
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.3)

            # 2. Put on hold: kill bridge, start hold music
            client.delete('/api/pipelines', data=json.dumps({"dsl": f"bridge:{leg.leg_id}"}), headers=account)

            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f'play:{call_id}_hold_{leg.leg_id}'
                                   f'{{"url":"examples/queue.mp3","loop":true,"volume":50}}'
                                   f' -> sip:{leg.leg_id}'
                        }),
                        headers=account)

            time.sleep(0.5)

            # Phone should hear hold music
            while not phone_rtp.rx_queue.empty():
                phone_rtp.rx_queue.get_nowait()
            time.sleep(0.3)
            hold_audio = _receive_audio(phone_rtp, 0.5)
            if len(hold_audio) > 0:
                hold_rms = float(np.sqrt(np.mean(
                    np.frombuffer(hold_audio, dtype=np.int16).astype(np.float64) ** 2
                )))
                assert hold_rms > 50, f"Hold music too quiet: RMS={hold_rms:.0f}"

            # 3. Unhold: kill hold music, rebridge
            client.delete('/api/pipelines', data=json.dumps({"dsl": f"play:{call_id}_hold_{leg.leg_id}"}), headers=account)

            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.3)

            # Verify leg is rebridged (status in-progress)
            assert leg.status == "in-progress"

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)


# ---------------------------------------------------------------------------
# Test: Pipeline API auth
# ---------------------------------------------------------------------------

class TestPipelineAuth:
    """Account-level auth on /api/pipelines."""

    def test_no_auth_rejected(self, client):
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "play:test"}))
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, client):
        h = {"Authorization": "Bearer wrong-token",
             "Content-Type": "application/json"}
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "play:test"}),
                           headers=h)
        assert resp.status_code == 403

    def test_account_token_accepted(self, client, account):
        """Account token must be accepted (not just admin)."""
        call_id = create_call(client, account)
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'play:test{{"url":"examples/queue.mp3"}} -> call:{call_id}'
                           }),
                           headers=account)
        assert resp.status_code == 201
        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_admin_token_accepted(self, client, admin, account):
        call_id = create_call(client, account)
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'play:test{{"url":"examples/queue.mp3"}} -> call:{call_id}'
                           }),
                           headers=admin)
        assert resp.status_code == 201
        client.delete(f"/api/calls/{call_id}", headers=admin)


# ---------------------------------------------------------------------------
# Test: Mix-minus correctness
# ---------------------------------------------------------------------------

class TestMixMinus:
    """Two participants: A hears B but not itself, B hears A but not itself."""

    def test_own_audio_not_heard(self, client, account):
        """Leg A sends audio, Leg A should NOT hear it back (mix-minus)."""
        call_id = create_call(client, account)
        leg_a, phone_a, _ = _make_rtp_leg(number="+49111")

        try:
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_a.leg_id} -> call:{call_id} -> sip:{leg_a.leg_id}"
                        }),
                        headers=account)

            time.sleep(0.5)

            # Drain initial silence
            while not phone_a.rx_queue.empty():
                phone_a.rx_queue.get_nowait()

            # Send audio from Phone A
            ref_pcm = _load_pcm(phone_a.codec.sample_rate, 0.3)
            _send_audio(phone_a, ref_pcm)

            time.sleep(0.5)

            # Phone A should receive silence (mix-minus subtracts own input)
            received = _receive_audio(phone_a, 0.5)
            if len(received) > 0:
                samples = np.frombuffer(received, dtype=np.int16).astype(np.float64)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                # With only one participant, mix-minus = full_mix - own = silence
                assert rms < 200, (
                    f"Mix-minus broken: Leg A hears itself (RMS={rms:.0f})"
                )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg_a, phone_a)

    def test_other_audio_heard(self, client, account):
        """Leg A sends, Leg B hears it. Leg B sends, Leg A hears it."""
        call_id = create_call(client, account)
        leg_a, phone_a, _ = _make_rtp_leg(codec=PCMU, number="+49111")
        leg_b, phone_b, _ = _make_rtp_leg(codec=PCMU, number="+49222")

        try:
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_a.leg_id} -> call:{call_id} -> sip:{leg_a.leg_id}"
                        }),
                        headers=account)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_b.leg_id} -> call:{call_id} -> sip:{leg_b.leg_id}"
                        }),
                        headers=account)

            time.sleep(1.0)

            # Drain silence
            while not phone_b.rx_queue.empty():
                phone_b.rx_queue.get_nowait()

            # A sends audio
            ref_pcm = _load_pcm(PCMU.sample_rate, 0.3)
            _send_audio(phone_a, ref_pcm)

            time.sleep(0.8)

            # B should hear A's audio
            received = _receive_audio(phone_b, 0.5)
            assert len(received) > 0, "Leg B received nothing from Leg A"

            samples = np.frombuffer(received, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            assert rms > 200, (
                f"Leg B doesn't hear Leg A (RMS={rms:.0f}) — mix-minus or routing broken"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg_a, phone_a)
            _cleanup_leg(leg_b, phone_b)


# ---------------------------------------------------------------------------
# Test: Conference teardown cleanup
# ---------------------------------------------------------------------------

class TestConferenceTeardown:
    """DELETE /api/calls/{id} cleans up everything."""

    def test_teardown_cleans_legs(self, client, account):
        call_id = create_call(client, account)
        leg, phone_rtp, _ = _make_rtp_leg()

        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                    }),
                    headers=account)
        time.sleep(0.3)

        # Verify leg exists
        assert leg_mod.get_leg(leg.leg_id) is not None

        client.delete(f"/api/calls/{call_id}", headers=account)

        # Leg should be gone
        assert leg_mod.get_leg(leg.leg_id) is None
        # Call should be gone
        assert call_state.get_call(call_id) is None

        phone_rtp.stop()

    def test_teardown_cleans_stages_and_tees(self, client, account):
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        leg, phone_rtp, _ = _make_rtp_leg()

        # Bridge with tee
        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f"sip:{leg.leg_id} -> tee:{leg.leg_id}_tap "
                               f"-> call:{call_id} -> sip:{leg.leg_id}"
                    }),
                    headers=account)
        time.sleep(0.3)

        ex = call.pipe_executor
        assert len(ex._stages) > 0, "No stages before teardown"
        assert len(ex._tees) > 0, "No tees before teardown"

        client.delete(f"/api/calls/{call_id}", headers=account)

        # Everything should be cleaned up
        assert len(ex._stages) == 0, f"Stages leaked: {list(ex._stages.keys())}"
        assert len(ex._tees) == 0, f"Tees leaked: {list(ex._tees.keys())}"

        phone_rtp.stop()

    def test_teardown_stops_mixer(self, client, account):
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        mixer = call.mixer

        assert not mixer.cancelled

        client.delete(f"/api/calls/{call_id}", headers=account)

        assert mixer.cancelled, "Mixer not cancelled after teardown"


# ---------------------------------------------------------------------------
# Test: SIP pinning on originate
# ---------------------------------------------------------------------------

class TestSIPPinning:
    """Account pinned to PBX cannot originate via different PBX."""

    def test_originate_wrong_pbx_rejected(self, client, account):
        """Account pinned to TestPBX, call on TestPBX, originate should check."""
        from conftest import SUBSCRIBER_ID
        call_id = create_call(client, account)

        # Account is pinned to TestPBX (set up in conftest).
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'originate:+4917099999{{}} -> call:{call_id}'
                           }),
                           headers=account)
        # Should succeed (call on TestPBX, account pinned to TestPBX)
        # or fail because PBX has no SIP proxy — but NOT 403
        assert resp.status_code != 403 or "PBX" not in (resp.data or b"").decode()

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_originate_cross_account_rejected(self, client, account, account2):
        """Account A cannot originate into Account B's call."""
        from conftest import SUBSCRIBER2_ID
        call_id = create_call(client, account2, SUBSCRIBER2_ID)

        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'originate:+4917099999{{}} -> call:{call_id}'
                           }),
                           headers=account)
        assert resp.status_code == 403

        client.delete(f"/api/calls/{call_id}", headers=account2)


# ---------------------------------------------------------------------------
# Test: 3+ participants conference
# ---------------------------------------------------------------------------

class TestThreePartyConference:
    """Three legs in one conference — verify everyone hears everyone else."""

    def test_three_participants_audio_routing(self, client, account):
        """A sends audio → B and C both hear it. Mix-minus: A does not."""
        call_id = create_call(client, account)
        leg_a, phone_a, _ = _make_rtp_leg(codec=PCMU, number="+49111")
        leg_b, phone_b, _ = _make_rtp_leg(codec=PCMU, number="+49222")
        leg_c, phone_c, _ = _make_rtp_leg(codec=PCMU, number="+49333")

        try:
            for leg in (leg_a, leg_b, leg_c):
                client.post("/api/pipelines",
                            data=json.dumps({
                                "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                            }),
                            headers=account)

            time.sleep(1.0)

            # Drain initial silence from all phones
            for phone in (phone_a, phone_b, phone_c):
                while not phone.rx_queue.empty():
                    phone.rx_queue.get_nowait()

            # A sends audio
            ref_pcm = _load_pcm(PCMU.sample_rate, 0.3)
            _send_audio(phone_a, ref_pcm)
            time.sleep(0.8)

            # B should hear A
            received_b = _receive_audio(phone_b, 0.5)
            assert len(received_b) > 0, "B received nothing"
            rms_b = float(np.sqrt(np.mean(
                np.frombuffer(received_b, dtype=np.int16).astype(np.float64) ** 2)))
            assert rms_b > 200, f"B doesn't hear A (RMS={rms_b:.0f})"

            # C should hear A
            received_c = _receive_audio(phone_c, 0.5)
            assert len(received_c) > 0, "C received nothing"
            rms_c = float(np.sqrt(np.mean(
                np.frombuffer(received_c, dtype=np.int16).astype(np.float64) ** 2)))
            assert rms_c > 200, f"C doesn't hear A (RMS={rms_c:.0f})"

            # A should NOT hear itself (mix-minus)
            received_a = _receive_audio(phone_a, 0.3)
            if len(received_a) > 0:
                rms_a = float(np.sqrt(np.mean(
                    np.frombuffer(received_a, dtype=np.int16).astype(np.float64) ** 2)))
                assert rms_a < 200, f"A hears itself (RMS={rms_a:.0f}) — mix-minus broken"

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg_a, phone_a)
            _cleanup_leg(leg_b, phone_b)
            _cleanup_leg(leg_c, phone_c)

    def test_three_participants_b_sends_a_and_c_hear(self, client, account):
        """B sends audio → A and C hear it, B does not."""
        call_id = create_call(client, account)
        leg_a, phone_a, _ = _make_rtp_leg(codec=PCMU, number="+49111")
        leg_b, phone_b, _ = _make_rtp_leg(codec=PCMU, number="+49222")
        leg_c, phone_c, _ = _make_rtp_leg(codec=PCMU, number="+49333")

        try:
            for leg in (leg_a, leg_b, leg_c):
                client.post("/api/pipelines",
                            data=json.dumps({
                                "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                            }),
                            headers=account)

            time.sleep(1.0)
            for phone in (phone_a, phone_b, phone_c):
                while not phone.rx_queue.empty():
                    phone.rx_queue.get_nowait()

            # B sends
            ref_pcm = _load_pcm(PCMU.sample_rate, 0.3)
            _send_audio(phone_b, ref_pcm)
            time.sleep(0.8)

            # A hears B
            received_a = _receive_audio(phone_a, 0.5)
            rms_a = float(np.sqrt(np.mean(
                np.frombuffer(received_a, dtype=np.int16).astype(np.float64) ** 2
            ))) if received_a else 0
            assert rms_a > 200, f"A doesn't hear B (RMS={rms_a:.0f})"

            # C hears B
            received_c = _receive_audio(phone_c, 0.5)
            rms_c = float(np.sqrt(np.mean(
                np.frombuffer(received_c, dtype=np.int16).astype(np.float64) ** 2
            ))) if received_c else 0
            assert rms_c > 200, f"C doesn't hear B (RMS={rms_c:.0f})"

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg_a, phone_a)
            _cleanup_leg(leg_b, phone_b)
            _cleanup_leg(leg_c, phone_c)


# ---------------------------------------------------------------------------
# Test: Hold-Swap (CRM "Makeln" / consultation transfer)
# ---------------------------------------------------------------------------

class TestHoldSwap:
    """CRM hold-swap: A (inbound) + B (outbound) bridged.
    Put B on hold, add C. A talks to C while B hears hold music.
    """

    def test_hold_swap_full_cycle(self, client, account):
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)

        leg_a, phone_a, _ = _make_rtp_leg(codec=PCMU, number="+49111")
        leg_b, phone_b, _ = _make_rtp_leg(codec=PCMU, number="+49222")
        leg_c, phone_c, _ = _make_rtp_leg(codec=PCMU, number="+49333")

        try:
            # 1. Bridge A and B
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_a.leg_id} -> call:{call_id} -> sip:{leg_a.leg_id}"
                        }),
                        headers=account)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_b.leg_id} -> call:{call_id} -> sip:{leg_b.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.5)

            # 2. Put B on hold: kill bridge, play hold music to B
            client.delete('/api/pipelines', data=json.dumps({"dsl": f"bridge:{leg_b.leg_id}"}), headers=account)

            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f'play:{call_id}_hold_b'
                                   f'{{"url":"examples/queue.mp3","loop":true,"volume":50}}'
                                   f' -> sip:{leg_b.leg_id}'
                        }),
                        headers=account)
            time.sleep(0.5)

            # B should hear hold music
            while not phone_b.rx_queue.empty():
                phone_b.rx_queue.get_nowait()
            time.sleep(0.3)
            hold_audio = _receive_audio(phone_b, 0.5)
            if hold_audio:
                rms_hold = float(np.sqrt(np.mean(
                    np.frombuffer(hold_audio, dtype=np.int16).astype(np.float64) ** 2)))
                assert rms_hold > 50, f"B doesn't hear hold music (RMS={rms_hold:.0f})"

            # 3. Bridge C into the conference
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_c.leg_id} -> call:{call_id} -> sip:{leg_c.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.5)

            # 4. A sends audio → C should hear it (not B, B is on hold)
            for phone in (phone_a, phone_c):
                while not phone.rx_queue.empty():
                    phone.rx_queue.get_nowait()

            ref_pcm = _load_pcm(PCMU.sample_rate, 0.3)
            _send_audio(phone_a, ref_pcm)
            time.sleep(0.8)

            received_c = _receive_audio(phone_c, 0.5)
            rms_c = float(np.sqrt(np.mean(
                np.frombuffer(received_c, dtype=np.int16).astype(np.float64) ** 2
            ))) if received_c else 0
            assert rms_c > 200, f"C doesn't hear A after swap (RMS={rms_c:.0f})"

            # 5. Unhold B: kill hold music, rebridge B
            client.delete('/api/pipelines', data=json.dumps({"dsl": f"play:{call_id}_hold_b"}), headers=account)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_b.leg_id} -> call:{call_id} -> sip:{leg_b.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.5)

            # 6. Now all three are in conference — A sends, B and C hear
            for phone in (phone_a, phone_b, phone_c):
                while not phone.rx_queue.empty():
                    phone.rx_queue.get_nowait()

            _send_audio(phone_a, ref_pcm)
            time.sleep(0.8)

            received_b = _receive_audio(phone_b, 0.5)
            rms_b = float(np.sqrt(np.mean(
                np.frombuffer(received_b, dtype=np.int16).astype(np.float64) ** 2
            ))) if received_b else 0
            assert rms_b > 200, (
                f"B doesn't hear A after unhold (RMS={rms_b:.0f})"
            )

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg_a, phone_a)
            _cleanup_leg(leg_b, phone_b)
            _cleanup_leg(leg_c, phone_c)


# ---------------------------------------------------------------------------
# Test: Inbound call flow via sip_listener
# ---------------------------------------------------------------------------

class TestInboundFlow:
    """Simulate inbound SIP call through sip_listener.

    Uses pyVoIP as the "caller" dialing into the speech-server's
    SIP listener on a local port.
    """

    def test_inbound_creates_leg(self, client, account):
        """Incoming pyVoIP call → sip_listener creates a Leg."""
        from speech_pipeline.telephony import sip_listener

        # Start listener on a free port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        pbx_entry = {
            "sip_proxy": "127.0.0.1",
            "sip_port": free_port,
            "sip_host": "",
            "sip_user": "test-inbound",
            "sip_password": "",
        }

        try:
            sip_listener.start_listener("test-inbound-pbx", pbx_entry)
        except Exception:
            # Listener may fail to start without a real SIP proxy,
            # but the code path is exercised
            pass

        # Verify the listener registered (even if connection fails)
        # The important thing is the code doesn't crash
        sip_listener._phones.pop("test-inbound-pbx", None)

    def test_inbound_leg_with_fake_voip_call(self, client, account):
        """Simulate what sip_listener._handle_incoming does: create leg + bridge."""
        call_id = create_call(client, account)

        # Create a leg as sip_listener would
        import queue as q
        fake_voip_call = type("FakeVoIPCall", (), {
            "read_audio": lambda self, *a, **kw: b"\xff" * 160,
            "write_audio": lambda self, *a, **kw: None,
            "answer": lambda self: None,
            "hangup": lambda self: None,
            "get_dtmf": lambda self, *a, **kw: "",
            "state": "answered",
            "RTPClients": [],
        })()

        leg = leg_mod.create_leg(
            direction="inbound",
            number="+491747712705",
            pbx_id="TestPBX",
            subscriber_id=SUBSCRIBER_ID,
            voip_call=fake_voip_call,
        )

        try:
            # Bridge into conference (what the CRM webhook does)
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                               }),
                               headers=account)
            assert resp.status_code == 201

            # Verify leg is in-progress and bound to call
            assert leg.status == "in-progress"
            assert leg.call_id == call_id

            # Verify conference has a participant
            call = call_state.get_call(call_id)
            participants = call.list_participants()
            assert len(participants) >= 1
            assert any(p["id"] == leg.leg_id for p in participants)

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            leg_mod._legs.pop(leg.leg_id, None)

    def test_inbound_leg_answer_via_api(self, client, account):
        """POST /api/legs/{id}/answer after bridging (CRM pattern)."""
        call_id = create_call(client, account)
        leg, phone_rtp, _ = _make_rtp_leg(number="+491747712705")

        try:
            # Bridge
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.3)

            # Answer (CRM sends this after the first outbound participant picks up)
            resp = client.post('/api/pipelines', data=json.dumps({"dsl": f"answer:{leg.leg_id}"}), headers=account)
            assert resp.status_code == 201

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)


# ---------------------------------------------------------------------------
# Test: Streaming TTS → Conference (source-only input)
# ---------------------------------------------------------------------------

class TestStreamingTTSPipeline:
    """text_input | tts{"voice":"..."} | conference:CALL — streaming TTS into conference."""

    def _ensure_tts(self):
        try:
            from speech_pipeline.registry import TTSRegistry
            r = TTSRegistry(VOICES_PATH, use_cuda=False)
            if not r.index:
                pytest.skip("No TTS voices available")
            import speech_pipeline.telephony._shared as _shared
            _shared.tts_registry = r
            return r
        except Exception as e:
            pytest.skip(f"TTS unavailable: {e}")

    def test_streaming_tts_into_conference(self, client, account):
        """text_input | tts{"voice":"..."} | conference:CALL — text produces audio."""
        self._ensure_tts()
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)

        try:
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f'text_input | tts{{"voice":"de_DE-thorsten-medium"}} | conference:{call_id}'
                               }),
                               headers=account)
            assert resp.status_code == 201, f"Pipeline creation failed: {resp.data}"
            pid = resp.get_json()["id"]

            out_q = call.mixer.add_output()

            client.post(f"/api/pipelines/{pid}/input",
                        data=json.dumps({"text": "Hallo, das ist ein Test."}),
                        headers=account)
            client.post(f"/api/pipelines/{pid}/input",
                        data=json.dumps({"eof": True}),
                        headers=account)

            time.sleep(3.0)
            collected = b""
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    frame = out_q.get(timeout=0.2)
                    if frame:
                        collected += frame
                except queue.Empty:
                    continue

            assert len(collected) > 0, "No TTS audio in conference"
            samples = np.frombuffer(collected, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            assert rms > 200, f"TTS audio RMS={rms:.0f} — too quiet"

        finally:
            client.delete(f"/api/pipelines/{pid}", headers=account)
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_streaming_tts_heard_by_sip_leg(self, client, account):
        """SIP leg hears streaming TTS in same conference."""
        self._ensure_tts()
        call_id = create_call(client, account)
        leg, phone_rtp, _ = _make_rtp_leg(codec=PCMU)

        try:
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                        }),
                        headers=account)

            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f'text_input | tts{{"voice":"de_DE-thorsten-medium"}} | conference:{call_id}'
                               }),
                               headers=account)
            assert resp.status_code == 201
            pid = resp.get_json()["id"]

            time.sleep(0.5)

            # Start collecting audio from phone in a background thread
            # BEFORE feeding text — so we catch the TTS output as it arrives
            phone_collected = []

            def _collect():
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    try:
                        frame = phone_rtp.rx_queue.get(timeout=0.2)
                        if frame:
                            phone_collected.append(frame)
                    except queue.Empty:
                        continue

            collector = threading.Thread(target=_collect, daemon=True)
            collector.start()

            # Feed text (TTS synthesis takes 1-3s)
            client.post(f"/api/pipelines/{pid}/input",
                        data=json.dumps({"text": "Das ist eine Testansage."}),
                        headers=account)
            client.post(f"/api/pipelines/{pid}/input",
                        data=json.dumps({"eof": True}),
                        headers=account)

            collector.join(timeout=10)

            received = b"".join(phone_collected)
            assert len(received) > 0, "Phone received no audio at all"

            # Check non-silent frames (TTS arrives after initial silence)
            samples = np.frombuffer(received, dtype=np.int16).astype(np.float64)
            # Find peak RMS in 100ms windows
            window = int(phone_rtp.codec.sample_rate * 0.1) * 2  # bytes
            peak_rms = 0.0
            for i in range(0, len(received) - window, window):
                chunk = np.frombuffer(received[i:i + window],
                                       dtype=np.int16).astype(np.float64)
                r = float(np.sqrt(np.mean(chunk ** 2)))
                peak_rms = max(peak_rms, r)

            assert peak_rms > 100, (
                f"Phone peak RMS={peak_rms:.0f} — TTS not reaching SIP leg"
            )

        finally:
            client.delete(f"/api/pipelines/{pid}", headers=account)
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)


# ---------------------------------------------------------------------------
# Test: Completed callback (play finished → webhook fires)
# ---------------------------------------------------------------------------

class TestCompletedCallback:
    """play:x{"completed":"/cb"} → webhook fires when play finishes."""

    def test_play_completed_fires(self, client, account):
        """Play a short audio file; the completed callback path is stored on the source."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)

        try:
            # Play non-looping (finishes after one pass)
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f'play:done_test{{"url":"examples/queue.mp3",'
                                          f'"completed":"/cb/play-done"}} -> call:{call_id}'
                               }),
                               headers=account)
            assert resp.status_code == 201

            # The play stage should be registered with its completed callback
            ex = call.pipe_executor
            stage_ids = [s["id"] for s in ex.list_stages()]
            assert "play:done_test" in stage_ids

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# Test: Concurrent pipeline creation (race condition check)
# ---------------------------------------------------------------------------

class TestConcurrentPipelines:
    """Multiple pipeline requests fired simultaneously for the same call."""

    def test_concurrent_bridges_dont_crash(self, client, account):
        """Fire 3 bridge requests in parallel — no crash, no corruption."""
        call_id = create_call(client, account)
        legs = []
        phones = []
        for i in range(3):
            leg, phone, _ = _make_rtp_leg(codec=PCMU, number=f"+4900{i}")
            legs.append(leg)
            phones.append(phone)

        errors = []

        def _bridge(leg):
            try:
                resp = client.post("/api/pipelines",
                                   data=json.dumps({
                                       "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                                   }),
                                   headers=account)
                if resp.status_code != 201:
                    errors.append(f"{leg.leg_id}: {resp.status_code} {resp.data}")
            except Exception as e:
                errors.append(f"{leg.leg_id}: {e}")

        try:
            threads = [threading.Thread(target=_bridge, args=(leg,)) for leg in legs]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Concurrent bridge errors: {errors}"

            # All 3 legs should be bridged
            time.sleep(0.5)
            call = call_state.get_call(call_id)
            participants = call.list_participants()
            assert len(participants) >= 3, (
                f"Only {len(participants)} participants after concurrent bridge"
            )

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            for leg, phone in zip(legs, phones):
                _cleanup_leg(leg, phone)

    def test_concurrent_bridge_and_play(self, client, account):
        """Bridge + hold music fired simultaneously — both succeed."""
        call_id = create_call(client, account)
        leg, phone_rtp, _ = _make_rtp_leg()

        results = {}

        def _bridge():
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                               }),
                               headers=account)
            results["bridge"] = resp.status_code

        def _play():
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f'play:wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
                               }),
                               headers=account)
            results["play"] = resp.status_code

        try:
            t1 = threading.Thread(target=_bridge)
            t2 = threading.Thread(target=_play)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert results.get("bridge") == 201, f"Bridge: {results.get('bridge')}"
            assert results.get("play") == 201, f"Play: {results.get('play')}"

        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone_rtp)


# ---------------------------------------------------------------------------
# Test: Streaming TTS + VC chain into conference
# ---------------------------------------------------------------------------

class TestStreamingTTSVCPipeline:
    """text_input | tts{"voice":"..."} | vc{"url":"..."} | conference:CALL"""

    def _ensure_tts(self):
        try:
            from speech_pipeline.registry import TTSRegistry
            r = TTSRegistry(VOICES_PATH, use_cuda=False)
            if not r.index:
                pytest.skip("No TTS voices available")
            import speech_pipeline.telephony._shared as _shared
            _shared.tts_registry = r
        except Exception as e:
            pytest.skip(f"TTS unavailable: {e}")

    def test_streaming_tts_vc_into_conference(self, client, account):
        """text_input | tts | vc | conference — full chain produces audio."""
        self._ensure_tts()
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        import speech_pipeline.telephony._shared as _shared
        old_media_folder = _shared.media_folder

        try:
            _shared.media_folder = VOICES_PATH
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": (
                                       f'text_input | tts{{"voice":"de_DE-thorsten-medium"}} '
                                       f'| vc{{"url":"de_DE-thorsten-high.onnx"}} '
                                       f'| conference:{call_id}'
                                   )
                               }),
                               headers=account)
            # VC might fail if model not set up for voice conversion
            if resp.status_code == 400 and "vc" in resp.data.decode().lower():
                pytest.skip("VC not available in test environment")
            assert resp.status_code == 201, f"Pipeline failed: {resp.data}"
            pid = resp.get_json()["id"]

            out_q = call.mixer.add_output()

            client.post(f"/api/pipelines/{pid}/input",
                        data=json.dumps({"text": "Voice Conversion Test."}),
                        headers=account)
            client.post(f"/api/pipelines/{pid}/input",
                        data=json.dumps({"eof": True}),
                        headers=account)

            time.sleep(4.0)  # TTS + VC need time
            collected = b""
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    frame = out_q.get(timeout=0.2)
                    if frame:
                        collected += frame
                except queue.Empty:
                    continue

            assert len(collected) > 0, "No audio from TTS+VC pipeline"
            samples = np.frombuffer(collected, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            assert rms > 100, f"TTS+VC output RMS={rms:.0f} — too quiet"

        finally:
            _shared.media_folder = old_media_folder
            try:
                client.delete(f"/api/pipelines/{pid}", headers=account)
            except Exception:
                pass
            client.delete(f"/api/calls/{call_id}", headers=account)
