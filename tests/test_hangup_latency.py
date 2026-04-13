"""Latency guards on the call-tear-down path.

Today's incidents: a webclient red-handset click took ~30 s to reach
the peer (mixer idle-timeout) instead of being instant.  These tests
enforce hard upper bounds on every leave-the-call path so a future
regression that removes the cleanup hook gets caught immediately.
"""
from __future__ import annotations

import json
import time

import pytest

from conftest import create_call
from speech_pipeline.telephony import auth as auth_mod
from speech_pipeline.telephony import call_state, webclient


HANGUP_LATENCY_BUDGET_S = 1.0


class TestWebclientHangupLatency:

    def test_phone_event_completed_detaches_within_1s(
            self, client, account):
        """Browser red handset → peer should be alone in <1 s, not
        wait for the mixer's 30 s idle timeout."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        try:
            nonce_entry = auth_mod.create_nonce(
                account_id=call.account_id,
                subscriber_id=call.subscriber_id,
                user="u1",
            )
            nonce = nonce_entry["nonce"]
            sess = webclient.register_webclient(call, "u1", nonce)
            sid = sess["session_id"]
            call.register_participant(sid, type="webclient", user="u1",
                                       nonce=nonce, callback="")

            t0 = time.monotonic()
            resp = client.post(
                f"/phone/{nonce}/event",
                data=json.dumps({"session": sid, "event": "completed"}),
                headers={"Content-Type": "application/json"},
            )
            elapsed = time.monotonic() - t0
            assert resp.status_code == 200
            assert elapsed < HANGUP_LATENCY_BUDGET_S, (
                f"phone_event(completed) took {elapsed*1000:.0f}ms — "
                "tear-down path is slow"
            )
            # Participant gone from the call.
            assert call.get_participant(sid) is None
            # Session removed from the registry.
            assert webclient.get_webclient_session(sid) is None
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)


class TestDeleteCallLatency:
    """``DELETE /api/calls/<id>`` must complete fast — the CRM blocks
    on this synchronously when ending an inbound call."""

    def test_delete_call_returns_quickly(self, client, account):
        call_id = create_call(client, account)
        t0 = time.monotonic()
        resp = client.delete(f"/api/calls/{call_id}", headers=account)
        elapsed = time.monotonic() - t0
        assert resp.status_code in (200, 204)
        assert elapsed < HANGUP_LATENCY_BUDGET_S, (
            f"DELETE /api/calls took {elapsed*1000:.0f}ms — peer SIP "
            "BYE / leg.hangup is blocking the response"
        )

    def test_delete_call_with_webclient_participant(self, client, account):
        """Conference with a webclient slot — DELETE must still be fast."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        nonce_entry = auth_mod.create_nonce(
            account_id=call.account_id,
            subscriber_id=call.subscriber_id, user="u1",
        )
        webclient.register_webclient(call, "u1", nonce_entry["nonce"])

        t0 = time.monotonic()
        resp = client.delete(f"/api/calls/{call_id}", headers=account)
        elapsed = time.monotonic() - t0
        assert resp.status_code in (200, 204)
        assert elapsed < HANGUP_LATENCY_BUDGET_S, (
            f"DELETE with webclient participant took {elapsed*1000:.0f}ms"
        )
