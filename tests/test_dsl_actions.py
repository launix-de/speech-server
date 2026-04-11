"""Tests for DSL action elements: kill, answer, originate.

These elements are not pipeline stages — they execute side effects
(stop a stage, send SIP 200 OK, dial out) via the unified DSL.
"""
import json
import threading
import time

import pytest

from conftest import SUBSCRIBER_ID, create_call
from speech_pipeline.telephony import call_state, leg as leg_mod


# ---------------------------------------------------------------------------
# kill: action
# ---------------------------------------------------------------------------

class TestKillAction:
    """kill:STAGE_ID — stops a running stage."""

    def test_kill_existing_stage(self, client, account):
        """kill: removes a play stage."""
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)

        # Create a play stage
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'play:test_kill{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
                           }),
                           headers=account)
        assert resp.status_code == 201

        # Verify stage exists
        ex = call.pipe_executor
        stage_ids = [s["id"] for s in ex.list_stages()]
        assert "play:test_kill" in stage_ids

        # Kill it via DSL
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "kill:play:test_kill"}),
                           headers=account)
        assert resp.status_code == 201

        # Verify stage is gone
        stage_ids = [s["id"] for s in ex.list_stages()]
        assert "play:test_kill" not in stage_ids

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_kill_nonexistent_stage(self, client, account):
        """kill: with unknown stage ID fails."""
        call_id = create_call(client, account)

        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "kill:nonexistent_stage"}),
                           headers=account)
        assert resp.status_code == 400

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_kill_bridge_stage(self, client, account):
        """kill:bridge:LEG_ID — detaches a leg from conference."""
        call_id = create_call(client, account)

        # Create a fake leg and bridge it
        from test_crm_e2e import _make_rtp_leg, _cleanup_leg
        from speech_pipeline.rtp_codec import PCMU
        leg, phone_rtp, _ = _make_rtp_leg(codec=PCMU)

        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                    }),
                    headers=account)
        time.sleep(0.3)

        # Kill the bridge via DSL
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f"kill:bridge:{leg.leg_id}"
                           }),
                           headers=account)
        assert resp.status_code == 201

        client.delete(f"/api/calls/{call_id}", headers=account)
        _cleanup_leg(leg, phone_rtp)

    def test_kill_missing_id_fails(self, client, account):
        """kill without ID is a parse/validation error."""
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "kill"}),
                           headers=account)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# answer: action
# ---------------------------------------------------------------------------

class TestAnswerAction:
    """answer:LEG_ID — sends SIP 200 OK on an inbound leg."""

    def test_answer_existing_leg(self, client, account):
        """answer: on an existing leg succeeds."""
        # Create a fake inbound leg
        from test_crm_e2e import _make_rtp_leg, _cleanup_leg
        from speech_pipeline.rtp_codec import PCMU
        leg, phone_rtp, _ = _make_rtp_leg(codec=PCMU, number="+4989123456")

        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": f"answer:{leg.leg_id}"}),
                           headers=account)
        assert resp.status_code == 201

        _cleanup_leg(leg, phone_rtp)

    def test_answer_nonexistent_leg(self, client, account):
        """answer: on unknown leg fails."""
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "answer:leg-doesnotexist"}),
                           headers=account)
        assert resp.status_code == 400

    def test_answer_missing_id(self, client, account):
        """answer without ID is an error."""
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "answer"}),
                           headers=account)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# originate: element (pipeline stage, not just action)
# ---------------------------------------------------------------------------

class TestOriginateElement:
    """originate:NUMBER{callbacks} — dials out and bridges into conference.

    Note: actual SIP origination requires a running SIP stack which
    isn't available in unit tests. We test validation and error cases.
    """

    def test_originate_without_call_fails(self, client, account):
        """originate needs a call context."""
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": 'originate:+4917099999{"answered":"/cb"}'
                           }),
                           headers=account)
        # Should fail because no call: in DSL and no call context
        assert resp.status_code == 400

    def test_originate_missing_number_fails(self, client, account):
        """originate without number fails."""
        call_id = create_call(client, account)
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f"originate -> call:{call_id} -> originate"
                           }),
                           headers=account)
        assert resp.status_code == 400
        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_originate_no_pbx_fails(self, client, account):
        """originate fails if PBX has no SIP proxy configured."""
        call_id = create_call(client, account)
        # TestPBX has empty sip_proxy — originate will fail
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'originate:+4917099999{{"answered":"/cb"}} -> call:{call_id} -> originate:+4917099999'
                           }),
                           headers=account)
        # Should fail because PBX has no SIP stack or proxy
        assert resp.status_code == 400
        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_originate_parses_correctly(self):
        """DSL parser handles originate with JSON callbacks."""
        from speech_pipeline.dsl_parser import parse_dsl
        result = parse_dsl(
            'originate:+4917099999{"answered":"/cb/ans","completed":"/cb/done"}'
            ' -> call:call-xxx -> originate:+4917099999'
        )
        assert len(result) == 3
        assert result[0][0] == "originate"
        assert result[0][1] == "+4917099999"
        assert result[0][2]["answered"] == "/cb/ans"
        assert result[2][0] == "originate"


# ---------------------------------------------------------------------------
# DSL parser tests for new action elements
# ---------------------------------------------------------------------------

class TestActionDSLParsing:
    """Verify parse_dsl handles kill:, answer:, originate: correctly."""

    def test_parse_kill(self):
        from speech_pipeline.dsl_parser import parse_dsl
        result = parse_dsl("kill:play:hold_music")
        assert result == [("kill", "play:hold_music", {})]

    def test_parse_answer(self):
        from speech_pipeline.dsl_parser import parse_dsl
        result = parse_dsl("answer:leg-abc123")
        assert result == [("answer", "leg-abc123", {})]

    def test_parse_originate_with_callbacks(self):
        from speech_pipeline.dsl_parser import parse_dsl
        result = parse_dsl('originate:+491747712705{"caller_id":"+4935863190000"}')
        assert result[0][0] == "originate"
        assert result[0][1] == "+491747712705"
        assert result[0][2]["caller_id"] == "+4935863190000"
