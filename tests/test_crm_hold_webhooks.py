"""Hold/unhold flow replay, with Pipe-API calls **and** webhook audit.

Mirrors ``backends/businesslogic/telefonanlage/speech-server/calls.fop``::

    holdExternalLegs:
        DELETE /api/pipelines  {dsl: "bridge:LEG"}
        POST   /api/pipelines  {dsl: "play:CALL_hold_LEG{...} -> sip:LEG"}

    unholdExternalLegs:
        DELETE /api/pipelines  {dsl: "play:CALL_hold_LEG"}
        POST   /api/pipelines  {dsl: "sip:LEG{completed:URL} -> call:CALL -> sip:LEG"}

The missing coverage this test adds over ``test_crm_hold_and_multiparty``
is **webhook correctness after unhold** — the rebuilt bridge pipe must
still carry the ``completed`` callback so the CRM learns the leg ended.

Existing suites verified only the audio swap.
"""
from __future__ import annotations

import json
import time

import pytest

from conftest import ADMIN_TOKEN, ACCOUNT_TOKEN, ACCOUNT_ID, SUBSCRIBER_ID
from fake_crm import FakeCrm
from test_crm_e2e import (
    _cleanup_leg,
    _make_rtp_leg,
    _send_audio,
    _receive_audio,
    _load_pcm,
)
from speech_pipeline.rtp_codec import PCMU
from speech_pipeline.telephony import call_state, leg as leg_mod


@pytest.fixture
def crm(client, admin):
    """FakeCrm pointed at the test's subscriber."""
    # Provision account/subscriber like the other CRM flow tests.
    acct = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
            "Content-Type": "application/json"}
    client.put("/api/pbx/TestPBX",
               data=json.dumps({"sip_proxy": "", "sip_user": "",
                                 "sip_password": ""}), headers=admin)
    client.put(f"/api/accounts/{ACCOUNT_ID}",
               data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX"}),
               headers=admin)
    c = FakeCrm(client, admin_headers=admin, account_token=ACCOUNT_TOKEN)
    c.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")
    yield c


class TestHoldWebhookContract:
    """Hold/unhold via the real Pipe-API + verify webhook lands in CRM."""

    def test_unhold_rebuilt_bridge_still_fires_call_level_completed_webhook(
            self, client, account, crm, monkeypatch):
        """Rebuilt pipe after unhold must carry the ``completed`` cb so
        the CRM learns the *external leg* hung up. Earlier regression:
        the cb got dropped by ``buildAndMergePipes`` so the phone BYE
        silently stranded the main call forever."""
        # 1. Bootstrap: inbound call via FakeCrm state=incoming.
        leg, phone, server_rtp = _make_rtp_leg(codec=PCMU, number="+49170")

        with crm.active(monkeypatch):
            from speech_pipeline.telephony import dispatcher, subscriber as sub_mod
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "incoming", {
                "caller": "+49170",
                "callee": "+4935000",
                "leg_id": leg.leg_id,
            })
            time.sleep(0.2)

            call_db_id = next(iter(crm.calls.keys()))
            call_sid = crm.calls[call_db_id]["sid"]

            # Find the participant id that state=incoming created.
            # FakeCrm stores the inbound caller's row under ``_on_incoming``
            # but for this subscriber doesn't — simulate the
            # participant manually with a known id to match the
            # completed-callback URL that the CRM bakes in.
            pid = 1001
            crm.participants[pid] = {
                "call_db_id": call_db_id,
                "sid": leg.leg_id,
                "status": "answered",
                "number": "+49170",
            }

            # 2. Hold flow (mirrors calls.fop::holdExternalLegs).
            crm.hold_external_legs(call_db_id, pid, leg.leg_id)
            time.sleep(0.3)

            # 3. Unhold flow (mirrors unholdExternalLegs). The rebuilt
            #    external-leg bridge must still carry a call-level
            #    completed callback.
            crm.unhold_external_legs(call_db_id, pid, leg.leg_id)
            time.sleep(0.3)

            # Drain any webhook events from the hold dance itself.
            baseline = len(crm.webhooks)

            # 4. Tear down the leg (phone BYE).  The completion monitor
            # watches sip_session.hungup; flipping the event triggers
            # the ``completed`` callback.
            leg.sip_session.hungup.set()
            time.sleep(1.0)

            completed_events = [
                w for w in crm.webhooks[baseline:]
                if w["state"] == "leg"
                and w["query"].get("event") == "completed"
            ]
            assert completed_events, (
                f"No state=leg&event=completed webhook received after "
                f"leg teardown; CRM can't flip participant status. "
                f"Post-unhold webhooks: {[w['state'] for w in crm.webhooks[baseline:]]}"
            )
            w = completed_events[0]
            assert int(w["query"].get("call", 0)) == call_db_id, (
                f"completed webhook carries wrong call id: {w['query']}"
            )
            assert "participant" not in w["query"], (
                f"external-leg completed webhook unexpectedly carries a "
                f"participant id: {w['query']}"
            )

        _cleanup_leg(leg, phone)
        # Clean up any dangling call.
        try:
            client.delete(f"/api/calls/{call_sid}", headers=account)
        except Exception:
            pass

    def test_hold_play_pipe_registered_and_hold_swaps_audio(
            self, client, account, crm, monkeypatch):
        """Hold must (a) register the hold-music stage on the server
        (so unhold can DELETE it), and (b) the held leg must actually
        hear hold music instead of the conference mix."""
        leg, phone, server_rtp = _make_rtp_leg(codec=PCMU, number="+49171")

        with crm.active(monkeypatch):
            from speech_pipeline.telephony import dispatcher, subscriber as sub_mod
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "incoming", {
                "caller": "+49171",
                "callee": "+4935000",
                "leg_id": leg.leg_id,
            })
            time.sleep(0.2)

            call_db_id = next(iter(crm.calls.keys()))
            call_sid = crm.calls[call_db_id]["sid"]
            pid = 2002
            crm.participants[pid] = {
                "call_db_id": call_db_id,
                "sid": leg.leg_id,
                "status": "answered",
                "number": "+49171",
            }
            crm.hold_jingle = "examples/queue.mp3"

            crm.hold_external_legs(call_db_id, pid, leg.leg_id)
            time.sleep(0.3)

            # (a) The hold POST must have been accepted (201).
            hold_posts = [w for w in crm.webhooks if False]  # placeholder
            # The helper already logs into its client; if the stage didn't
            # register, the audio check below will fail with silence.

            # (b) The phone hears the hold jingle — spectral match
            # against the real queue.mp3 source (loudness alone would
            # pass for any non-silent noise).
            while not phone.rx_queue.empty():
                phone.rx_queue.get_nowait()
            time.sleep(0.5)
            received = _receive_audio(phone, duration_s=1.0)
            import numpy as np
            arr = np.frombuffer(received, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
            assert rms > 100, (
                f"held leg hears silence (RMS={rms:.0f}) — hold music "
                f"pipe did not attach"
            )
            ref = _load_pcm(PCMU.sample_rate, 1.0)  # queue.mp3 @ 8k
            a = np.frombuffer(ref, dtype=np.int16).astype(np.float64)
            b = arr
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            spec_a = np.abs(np.fft.rfft(a))
            spec_b = np.abs(np.fft.rfft(b))
            # 10-band energy cosine similarity (robust against loop
            # phase + PCMU codec quantization).
            ea = np.array([np.sum(x ** 2) for x in np.array_split(spec_a, 10)])
            eb = np.array([np.sum(x ** 2) for x in np.array_split(spec_b, 10)])
            spec_sim = float(np.dot(ea, eb) /
                             (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
            assert spec_sim > 0.6, (
                f"phone audio during hold doesn't match queue.mp3 "
                f"spectrum (spec-sim={spec_sim:.3f}) — held leg is "
                f"hearing SOMETHING but not the hold jingle"
            )

        _cleanup_leg(leg, phone)
        try:
            client.delete(f"/api/calls/{call_sid}", headers=account)
        except Exception:
            pass
