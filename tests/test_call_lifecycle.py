"""Tests for the full call lifecycle via the REST API.

Sequence (mirrors fop-dev webhook flow):
1. POST /api/calls → create conference
2. POST /api/calls/{id}/pipes → wire inbound leg + wait music
3. POST /api/legs/originate → ring outbound participant
4. POST /api/calls/{id}/pipes → bridge answered participant
5. DELETE /api/calls/{id}/stages/play:... → kill wait music
6. DELETE /api/calls/{id} → teardown, verify cleanup
"""
import json
import threading
import time

import pytest

from conftest import (
    ADMIN_TOKEN, ACCOUNT_TOKEN, SUBSCRIBER_ID,
    create_call,
)
from speech_pipeline.telephony import leg as leg_mod, call_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inbound_leg(client, headers, number="+4989123456"):
    """Register an inbound leg (simulates SIP listener creating one)."""
    leg = leg_mod.Leg(
        leg_id=f"leg-in-{number}",
        direction="inbound",
        number=number,
        pbx_id="TestPBX",
        subscriber_id=SUBSCRIBER_ID,
    )
    # Mock SIP session
    from unittest.mock import MagicMock
    mock_call = MagicMock()
    mock_call.read_audio.side_effect = Exception("stopped")
    mock_call.get_dtmf.return_value = ""
    leg.voip_call = mock_call

    from speech_pipeline.telephony.leg import PyVoIPCallSession
    session = PyVoIPCallSession(mock_call)
    leg.sip_session = session
    leg.status = "ringing"

    leg_mod._legs[leg.leg_id] = leg
    return leg


# ---------------------------------------------------------------------------
# Test: create and delete call
# ---------------------------------------------------------------------------

class TestCreateDeleteCall:

    def test_create_call(self, client, account):
        call_id = create_call(client, account)
        assert call_id.startswith("call-")

        # Call exists
        resp = client.get(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 200

    def test_delete_call(self, client, account):
        call_id = create_call(client, account)
        resp = client.delete(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 204

        # Call is gone
        resp = client.get(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 404

    def test_double_delete_returns_404(self, client, account):
        call_id = create_call(client, account)
        client.delete(f"/api/calls/{call_id}", headers=account)
        resp = client.delete(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test: pipe wait music into conference
# ---------------------------------------------------------------------------

class TestWaitMusic:

    def test_add_play_pipe(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(
            f"/api/calls/{call_id}/pipes",
            data=json.dumps({
                "pipes": [
                    f'play:{call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
                ]
            }),
            headers=account,
        )
        assert resp.status_code == 202
        results = resp.get_json()["results"]
        assert results[0]["ok"] is True

        # Stage is registered on the pipe executor
        call = call_state.get_call(call_id)
        assert call.pipe_executor is not None
        stage_ids = [s["id"] for s in call.pipe_executor.list_stages()]
        assert f"play:{call_id}_wait" in stage_ids

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_kill_play_stage(self, client, account):
        call_id = create_call(client, account)
        client.post(
            f"/api/calls/{call_id}/pipes",
            data=json.dumps({
                "pipes": [
                    f'play:{call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
                ]
            }),
            headers=account,
        )

        resp = client.delete(
            f"/api/calls/{call_id}/stages/play:{call_id}_wait",
            headers=account,
        )
        assert resp.status_code == 204

        # Stage is gone
        call = call_state.get_call(call_id)
        stage_ids = [s["id"] for s in call.pipe_executor.list_stages()]
        assert f"play:{call_id}_wait" not in stage_ids

        client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# Test: bridge inbound leg
# ---------------------------------------------------------------------------

class TestBridgeLeg:

    def test_bridge_inbound_leg(self, client, account):
        """Wire an inbound leg into the conference via DSL pipe."""
        call_id = create_call(client, account)
        leg = _make_inbound_leg(client, account)

        resp = client.post(
            f"/api/calls/{call_id}/pipes",
            data=json.dumps({
                "pipes": [
                    f'sip:{leg.leg_id}{{"completed":"/cb/done"}} -> call:{call_id} -> sip:{leg.leg_id}'
                ]
            }),
            headers=account,
        )
        assert resp.status_code == 202

        # Leg should be in-progress now
        assert leg.status == "in-progress"
        assert leg.call_id == call_id

        # Bridge handle registered
        call = call_state.get_call(call_id)
        stage_ids = [s["id"] for s in call.pipe_executor.list_stages()]
        assert f"bridge:{leg.leg_id}" in stage_ids

        # Cleanup
        client.delete(f"/api/calls/{call_id}", headers=account)
        leg_mod._legs.pop(leg.leg_id, None)

    def test_answer_inbound_leg(self, client, account):
        """POST /api/legs/{id}/answer sets leg status."""
        call_id = create_call(client, account)
        leg = _make_inbound_leg(client, account, "+4989999999")

        # Bridge first
        client.post(
            f"/api/calls/{call_id}/pipes",
            data=json.dumps({
                "pipes": [
                    f'sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}'
                ]
            }),
            headers=account,
        )

        resp = client.post(f"/api/legs/{leg.leg_id}/answer", headers=account)
        assert resp.status_code == 200

        client.delete(f"/api/calls/{call_id}", headers=account)
        leg_mod._legs.pop(leg.leg_id, None)


# ---------------------------------------------------------------------------
# Test: full lifecycle sequence
# ---------------------------------------------------------------------------

class TestFullLifecycle:

    def test_full_inbound_call_sequence(self, client, account):
        """Replicate the fop-dev inbound call flow end-to-end."""
        # 1. Create conference
        call_id = create_call(client, account)

        # 2. Wire inbound leg + wait music
        leg = _make_inbound_leg(client, account, "+4989111111")
        client.post(
            f"/api/calls/{call_id}/pipes",
            data=json.dumps({
                "pipes": [
                    f'sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}',
                    f'play:{call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}',
                ]
            }),
            headers=account,
        )

        # 3. Wait music is playing
        call = call_state.get_call(call_id)
        stages = [s["id"] for s in call.pipe_executor.list_stages()]
        assert f"play:{call_id}_wait" in stages
        assert f"bridge:{leg.leg_id}" in stages

        # 4. Simulate outbound answered: add second leg
        leg2 = _make_inbound_leg(client, account, "+491747712705")
        leg2.leg_id = "leg-out-1747"
        leg_mod._legs[leg2.leg_id] = leg2

        client.post(
            f"/api/calls/{call_id}/pipes",
            data=json.dumps({
                "pipes": [
                    f'sip:{leg2.leg_id} -> call:{call_id} -> sip:{leg2.leg_id}'
                ]
            }),
            headers=account,
        )

        # 5. Kill wait music
        resp = client.delete(
            f"/api/calls/{call_id}/stages/play:{call_id}_wait",
            headers=account,
        )
        assert resp.status_code == 204

        # 6. Answer inbound
        client.post(f"/api/legs/{leg.leg_id}/answer", headers=account)

        # 7. Both legs bridged
        stages = [s["id"] for s in call.pipe_executor.list_stages()]
        assert f"bridge:{leg.leg_id}" in stages
        assert f"bridge:{leg2.leg_id}" in stages
        assert f"play:{call_id}_wait" not in stages

        # 8. Teardown
        resp = client.delete(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 204

        # 9. Verify cleanup
        assert call_state.get_call(call_id) is None

        leg_mod._legs.pop(leg.leg_id, None)
        leg_mod._legs.pop(leg2.leg_id, None)
