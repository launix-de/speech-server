"""CRM hold / unhold flow + dynamic 3+ participant conference.

Mirrors the exact DSL sequence emitted by
``backends/businesslogic/telefonanlage/speech-server/calls.fop``:

* ``holdExternalLegs``::

      DELETE /api/pipelines  {dsl: "bridge:LEG"}
      DELETE /api/pipelines  {dsl: "play:CALL_hold_LEG"}   # idempotent
      POST   /api/pipelines  {dsl: 'play:CALL_hold_LEG{"url":...} -> sip:LEG'}

* ``unholdExternalLegs``::

      DELETE /api/pipelines  {dsl: "play:CALL_hold_LEG"}
      POST   /api/pipelines  {dsl: 'sip:LEG{cb} -> call:CALL -> sip:LEG'}

Every test uses a real ``RTPSession`` + audio-similarity/RMS check,
so codec path + mix-minus + teardown are all exercised end to end.
"""
from __future__ import annotations

import json
import queue
import time

import numpy as np
import pytest

from conftest import create_call
from speech_pipeline.rtp_codec import PCMU
from speech_pipeline.telephony import call_state

# Reuse the existing test helpers (no point duplicating).
from test_crm_e2e import (
    _cleanup_leg,
    _load_pcm,
    _make_rtp_leg,
    _receive_audio,
    _send_audio,
    _similarity,
    QUEUE_MP3,
)


# ---------------------------------------------------------------------------
# Hold / unhold with audio-quality check
# ---------------------------------------------------------------------------

class TestHoldAudioSimilarity:
    """When the CRM puts a leg on hold, the phone must hear hold music
    (not conference mix).  When unholded, conference audio resumes."""

    def _ref_hold_music(self, sample_rate: int, duration_s: float) -> bytes:
        """Load the hold-music source at the phone's sample rate."""
        return _load_pcm(sample_rate, duration_s)

    def test_hold_swaps_phone_audio_to_music(self, client, account):
        call_id = create_call(client, account)
        leg, phone, _ = _make_rtp_leg(codec=PCMU, number="+49111")

        try:
            # 1. Bridge phone → conference.  This is the "on-call" state.
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg.leg_id} -> call:{call_id} "
                                   f"-> sip:{leg.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.3)

            # Drain pre-hold silence from the phone's rx queue.
            while not phone.rx_queue.empty():
                phone.rx_queue.get_nowait()

            # 2. Hold: CRM drops the bridge and plays hold music straight
            #    into the leg.
            client.delete("/api/pipelines",
                          data=json.dumps({"dsl": f"bridge:{leg.leg_id}"}),
                          headers=account)
            hold_stage = f"play:{call_id}_hold_{leg.leg_id}"
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f'play:{call_id}_hold_{leg.leg_id}'
                                   f'{{"url":"examples/queue.mp3",'
                                   f'"loop":true,"volume":100}} '
                                   f'-> sip:{leg.leg_id}'
                        }),
                        headers=account)

            # 3. Collect a chunk of audio the phone receives.
            time.sleep(0.8)  # let hold music fill buffers
            while not phone.rx_queue.empty():
                phone.rx_queue.get_nowait()
            time.sleep(0.3)
            received = _receive_audio(phone, duration_s=1.0)
            assert len(received) > 0, "phone received nothing during hold"

            rms = float(np.sqrt(np.mean(
                np.frombuffer(received, dtype=np.int16).astype(np.float64) ** 2
            )))
            assert rms > 100, f"hold music too quiet: RMS={rms:.0f}"

            # Spectral match against the hold-music source.  Waveform
            # cross-correlation is unreliable because the loop phase +
            # PCMU codec loss; compare the energy distribution across
            # ten bands instead (robust against time offset).
            ref = self._ref_hold_music(PCMU.sample_rate, 1.0)
            a = np.frombuffer(ref, dtype=np.int16).astype(np.float64)
            b = np.frombuffer(received, dtype=np.int16).astype(np.float64)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            spec_a = np.abs(np.fft.rfft(a))
            spec_b = np.abs(np.fft.rfft(b))
            # Bucket into 10 frequency bands + cosine similarity.
            bands_a = np.array_split(spec_a, 10)
            bands_b = np.array_split(spec_b, 10)
            ea = np.array([np.sum(x ** 2) for x in bands_a])
            eb = np.array([np.sum(x ** 2) for x in bands_b])
            spec_sim = float(np.dot(ea, eb) /
                             (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
            assert spec_sim > 0.6, (
                f"Phone audio during hold does not match hold-music spectrum "
                f"(spectral-similarity={spec_sim:.3f})"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg, phone)

    def test_unhold_resumes_conference_audio(self, client, account):
        """Hold → unhold → phone hears audio from a second participant,
        not the hold-music loop."""
        call_id = create_call(client, account)
        leg_a, phone_a, _ = _make_rtp_leg(codec=PCMU, number="+49111")
        leg_b, phone_b, _ = _make_rtp_leg(codec=PCMU, number="+49222")

        try:
            # 1. Bridge A and B.
            for leg in (leg_a, leg_b):
                client.post("/api/pipelines",
                            data=json.dumps({
                                "dsl": f"sip:{leg.leg_id} -> call:{call_id} "
                                       f"-> sip:{leg.leg_id}"
                            }),
                            headers=account)
            time.sleep(0.3)

            # 2. Hold A: bridge → hold music.
            client.delete("/api/pipelines",
                          data=json.dumps({"dsl": f"bridge:{leg_a.leg_id}"}),
                          headers=account)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f'play:{call_id}_hold_{leg_a.leg_id}'
                                   f'{{"url":"examples/queue.mp3","loop":true}} '
                                   f'-> sip:{leg_a.leg_id}'
                        }),
                        headers=account)
            time.sleep(0.4)

            # 3. Unhold: drop hold music, re-bridge A.
            client.delete("/api/pipelines",
                          data=json.dumps({
                              "dsl": f"play:{call_id}_hold_{leg_a.leg_id}"
                          }),
                          headers=account)
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f"sip:{leg_a.leg_id} -> call:{call_id} "
                                   f"-> sip:{leg_a.leg_id}"
                        }),
                        headers=account)
            time.sleep(0.3)

            # Drain; then B sends audio; A must hear it.
            while not phone_a.rx_queue.empty():
                phone_a.rx_queue.get_nowait()

            pcm = _load_pcm(PCMU.sample_rate, 0.4)
            _send_audio(phone_b, pcm)
            time.sleep(0.8)

            received = _receive_audio(phone_a, duration_s=0.5)
            rms = float(np.sqrt(np.mean(
                np.frombuffer(received, dtype=np.int16).astype(np.float64) ** 2
            ))) if received else 0
            assert rms > 200, (
                f"A did not hear B after unhold (RMS={rms:.0f}) — "
                f"bridge not re-established"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            _cleanup_leg(leg_a, phone_a)
            _cleanup_leg(leg_b, phone_b)


# ---------------------------------------------------------------------------
# Dynamic participants — join, leave, rejoin
# ---------------------------------------------------------------------------

class TestDynamicParticipants:
    """Participants come and go mid-call.  The mix-minus must always
    be correct for whoever is in the conference *right now*."""

    def _bridge(self, client, headers, leg_id, call_id):
        return client.post(
            "/api/pipelines",
            data=json.dumps({
                "dsl": f"sip:{leg_id} -> call:{call_id} -> sip:{leg_id}"
            }),
            headers=headers,
        )

    def _unbridge(self, client, headers, leg_id):
        return client.delete(
            "/api/pipelines",
            data=json.dumps({"dsl": f"bridge:{leg_id}"}),
            headers=headers,
        )

    def test_four_participants_join_leave_rejoin(self, client, account):
        call_id = create_call(client, account)
        legs = [
            _make_rtp_leg(codec=PCMU, number=f"+4911{i}")
            for i in range(4)
        ]

        try:
            # A, B, C, D all join.
            for leg, phone, _ in legs:
                self._bridge(client, account, leg.leg_id, call_id)
            time.sleep(0.5)

            # B leaves.
            self._unbridge(client, account, legs[1][0].leg_id)
            time.sleep(0.3)

            # D leaves.
            self._unbridge(client, account, legs[3][0].leg_id)
            time.sleep(0.3)

            # Conference = A + C.  A sends audio; C hears it, B and D don't.
            for _leg, phone, _ in legs:
                while not phone.rx_queue.empty():
                    phone.rx_queue.get_nowait()
            pcm = _load_pcm(PCMU.sample_rate, 0.3)
            _send_audio(legs[0][1], pcm)  # A sends
            time.sleep(0.8)

            def _rms(phone):
                r = _receive_audio(phone, duration_s=0.4)
                if not r:
                    return 0.0
                return float(np.sqrt(np.mean(
                    np.frombuffer(r, dtype=np.int16).astype(np.float64) ** 2
                )))

            rms_a = _rms(legs[0][1])
            rms_b = _rms(legs[1][1])
            rms_c = _rms(legs[2][1])
            rms_d = _rms(legs[3][1])

            assert rms_c > 200, f"C (active) did not hear A: RMS={rms_c:.0f}"
            assert rms_b < 200, f"B (unbridged) still hears audio: RMS={rms_b:.0f}"
            assert rms_d < 200, f"D (unbridged) still hears audio: RMS={rms_d:.0f}"
            assert rms_a < 200, f"A hears itself (mix-minus broken): RMS={rms_a:.0f}"

            # Rejoin B.
            self._bridge(client, account, legs[1][0].leg_id, call_id)
            time.sleep(0.5)

            for _leg, phone, _ in legs:
                while not phone.rx_queue.empty():
                    phone.rx_queue.get_nowait()

            # C sends; A and B hear it now.
            _send_audio(legs[2][1], pcm)
            time.sleep(0.8)

            rms_a = _rms(legs[0][1])
            rms_b = _rms(legs[1][1])
            assert rms_a > 200, (
                f"A did not hear C after B rejoin: RMS={rms_a:.0f}"
            )
            assert rms_b > 200, (
                f"B (rejoined) did not hear C: RMS={rms_b:.0f}"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            for leg, phone, _ in legs:
                _cleanup_leg(leg, phone)

    def test_participant_leaves_does_not_break_others(self, client, account):
        """When one leg hangs up, the remaining legs keep working."""
        call_id = create_call(client, account)
        leg_a, phone_a, _ = _make_rtp_leg(codec=PCMU, number="+49111")
        leg_b, phone_b, _ = _make_rtp_leg(codec=PCMU, number="+49222")
        leg_c, phone_c, _ = _make_rtp_leg(codec=PCMU, number="+49333")

        try:
            for leg in (leg_a, leg_b, leg_c):
                self._bridge(client, account, leg.leg_id, call_id)
            time.sleep(0.5)

            # C leaves.
            self._unbridge(client, account, leg_c.leg_id)
            time.sleep(0.3)

            # A→B still works.
            for phone in (phone_a, phone_b):
                while not phone.rx_queue.empty():
                    phone.rx_queue.get_nowait()
            pcm = _load_pcm(PCMU.sample_rate, 0.3)
            _send_audio(phone_a, pcm)
            time.sleep(0.8)

            received = _receive_audio(phone_b, duration_s=0.4)
            rms = float(np.sqrt(np.mean(
                np.frombuffer(received, dtype=np.int16).astype(np.float64) ** 2
            ))) if received else 0.0
            assert rms > 200, (
                f"B did not hear A after C left: RMS={rms:.0f}"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
            for leg, phone in [(leg_a, phone_a), (leg_b, phone_b), (leg_c, phone_c)]:
                _cleanup_leg(leg, phone)
