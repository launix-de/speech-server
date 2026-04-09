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
from speech_pipeline.rtp_codec import PCMU, PCMA

QUEUE_MP3 = os.path.join(os.path.dirname(__file__), "..", "examples", "queue.mp3")
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
        time.sleep(0.3)  # let play stage start
        collected = b""
        deadline = time.monotonic() + 2.0
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
        resp = client.delete(
            f"/api/calls/{call_id}/stages/play:{call_id}_wait",
            headers=account)
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
