"""Tests for the full call lifecycle via the REST API.

Sequence (mirrors fop-dev webhook flow):
1. POST /api/calls → create conference
2. POST /api/pipelines → wire inbound leg + wait music
3. POST /api/legs/originate → ring outbound participant
4. POST /api/pipelines → bridge answered participant
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
        leg_id=f"leg-test-{number.replace('+', '')}",
        direction="inbound",
        number=number,
        pbx_id="TestPBX",
        subscriber_id=SUBSCRIBER_ID,
    )
    # Give it a fake voip_call so SIPSource/SIPSink don't crash
    import queue
    fake_call = type("FakeVoIPCall", (), {
        "read_audio": lambda self, *a, **kw: b"\xff" * 160,
        "write_audio": lambda self, *a, **kw: None,
        "get_dtmf": lambda self, *a, **kw: "",
        "state": "answered",
        "RTPClients": [],
    })()
    leg.voip_call = fake_call
    leg.sip_session = type("FakeSession", (), {
        "call": fake_call,
        "connected": threading.Event(),
        "hungup": threading.Event(),
        "rx_queue": queue.Queue(),
    })()
    leg.sip_session.connected.set()
    leg_mod._legs[leg.leg_id] = leg
    return leg


def _post_pipes(client, headers, dsl_strings):
    """Post one or more DSL strings as separate pipelines."""
    results = []
    for dsl in dsl_strings:
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": dsl}),
                           headers=headers)
        results.append(resp)
    return results


# ---------------------------------------------------------------------------
# Test: nonexistent call in DSL
# ---------------------------------------------------------------------------

class TestCallNotFound:

    def test_pipe_with_nonexistent_call(self, client, account):
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": "play:x -> call:call-doesnotexist"
                           }),
                           headers=account)
        assert resp.status_code in (400, 403)


# ---------------------------------------------------------------------------
# Test: pipe wait music into conference
# ---------------------------------------------------------------------------

class TestWaitMusic:

    def test_add_play_pipe(self, client, account):
        call_id = create_call(client, account)
        resps = _post_pipes(client, account, [
            f'play:{call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
        ])
        assert resps[0].status_code == 201

        # Stage is registered on the pipe executor
        call = call_state.get_call(call_id)
        assert call.pipe_executor is not None
        stage_ids = [s["id"] for s in call.pipe_executor.list_stages()]
        assert f"play:{call_id}_wait" in stage_ids

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_kill_play_stage(self, client, account):
        call_id = create_call(client, account)
        _post_pipes(client, account, [
            f'play:{call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
        ])

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

        resps = _post_pipes(client, account, [
            f'sip:{leg.leg_id}{{"completed":"/cb/done"}} -> call:{call_id} -> sip:{leg.leg_id}'
        ])
        assert resps[0].status_code == 201

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
        _post_pipes(client, account, [
            f'sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}'
        ])

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
        _post_pipes(client, account, [
            f'sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}',
            f'play:{call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}',
        ])

        # 3. Wait music is playing
        call = call_state.get_call(call_id)
        stages = [s["id"] for s in call.pipe_executor.list_stages()]
        assert f"play:{call_id}_wait" in stages
        assert f"bridge:{leg.leg_id}" in stages

        # 4. Simulate outbound answered: add second leg
        leg2 = _make_inbound_leg(client, account, "+491747712705")
        leg2.leg_id = "leg-out-1747"
        leg_mod._legs[leg2.leg_id] = leg2

        _post_pipes(client, account, [
            f'sip:{leg2.leg_id} -> call:{call_id} -> sip:{leg2.leg_id}'
        ])

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
