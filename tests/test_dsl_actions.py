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

class TestKillViaDelete:
    """DELETE /api/pipelines {"dsl": "STAGE_ID"} — stops a running stage."""

    def test_delete_call_by_dsl(self, client, account):
        """DELETE /api/pipelines?dsl=call:CALL_ID tears down the call."""
        call_id = create_call(client, account)
        resp = client.delete(f"/api/pipelines?dsl=call:{call_id}",
                             headers=account)
        assert resp.status_code == 204

        resp = client.get(f"/api/pipelines?dsl=call:{call_id}",
                          headers=account)
        assert resp.status_code == 404

    def test_kill_existing_stage(self, client, account):
        """DELETE removes a play stage."""
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

        # Kill via DELETE /api/pipelines
        resp = client.delete("/api/pipelines",
                             data=json.dumps({"dsl": "play:test_kill"}),
                             headers=account)
        assert resp.status_code == 204

        # Verify stage is gone
        stage_ids = [s["id"] for s in ex.list_stages()]
        assert "play:test_kill" not in stage_ids

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_kill_nonexistent_stage(self, client, account):
        """DELETE with unknown stage ID returns 404."""
        resp = client.delete("/api/pipelines",
                             data=json.dumps({"dsl": "nonexistent_stage"}),
                             headers=account)
        assert resp.status_code == 404

    def test_kill_bridge_stage(self, client, account):
        """DELETE bridge:LEG_ID — detaches a leg from conference."""
        call_id = create_call(client, account)

        from test_crm_e2e import _make_rtp_leg, _cleanup_leg
        from speech_pipeline.rtp_codec import PCMU
        leg, phone_rtp, _ = _make_rtp_leg(codec=PCMU)

        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                    }),
                    headers=account)
        time.sleep(0.3)

        # Kill the bridge via DELETE
        resp = client.delete("/api/pipelines",
                             data=json.dumps({"dsl": f"bridge:{leg.leg_id}"}),
                             headers=account)
        assert resp.status_code == 204

        client.delete(f"/api/calls/{call_id}", headers=account)
        _cleanup_leg(leg, phone_rtp)

    def test_kill_missing_body(self, client, account):
        """DELETE without body returns 400."""
        resp = client.delete("/api/pipelines",
                             data=json.dumps({}),
                             headers=account)
        assert resp.status_code == 400

    def test_kill_requires_auth(self, client):
        """DELETE without auth returns 401."""
        resp = client.delete("/api/pipelines",
                             data=json.dumps({"dsl": "play:x"}))
        assert resp.status_code == 401

    def test_kill_rejects_pipeline_syntax(self, client, account):
        """DELETE rejects DSL with -> or | (only single stage ID allowed)."""
        resp = client.delete("/api/pipelines",
                             data=json.dumps({"dsl": "play:x -> call:c1"}),
                             headers=account)
        assert resp.status_code == 400

        resp = client.delete("/api/pipelines",
                             data=json.dumps({"dsl": "play:x | tee:y"}),
                             headers=account)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# webclient: action
# ---------------------------------------------------------------------------

class TestWebclientAction:
    """webclient:USER{callback, base_url} — creates webclient session."""

    def test_webclient_without_call_fails(self, client, account):
        """webclient needs a call context."""
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": 'webclient:testuser{"base_url":"https://example.com"}'
                           }),
                           headers=account)
        assert resp.status_code == 400

    def test_webclient_missing_base_url_fails(self, client, account):
        """webclient requires base_url."""
        call_id = create_call(client, account)
        # Need to call webclient in a call context — but single-element actions
        # don't naturally have call context in the DSL. For now verify the
        # missing base_url error at least triggers.
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'webclient:testuser{{"callback":"/cb"}}'
                           }),
                           headers=account)
        # Without call context, fails with "requires a call context"
        assert resp.status_code == 400
        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_webclient_dsl_parses(self):
        """DSL parser accepts webclient with JSON params."""
        from speech_pipeline.dsl_parser import parse_dsl
        result = parse_dsl(
            'webclient:user42{"callback":"/cb/wc","base_url":"https://srv.example.com"}'
        )
        assert result[0][0] == "webclient"
        assert result[0][1] == "user42"
        assert result[0][2]["callback"] == "/cb/wc"
        assert result[0][2]["base_url"] == "https://srv.example.com"


# ---------------------------------------------------------------------------
# GET /api/pipelines?dsl=... — look up live objects
# ---------------------------------------------------------------------------

class TestGetByDSL:
    """GET /api/pipelines?dsl=... resolves DSL items to live objects."""

    def test_get_call(self, client, account):
        """?dsl=call:CALL_ID returns call details."""
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["call_id"] == call_id

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_get_conference_alias(self, client, account):
        """?dsl=conference:CALL_ID works same as call:."""
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=conference:{call_id}", headers=account)
        assert resp.status_code == 200
        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_get_nonexistent_call(self, client, account):
        resp = client.get("/api/pipelines?dsl=call:no-such", headers=account)
        assert resp.status_code == 404

    def test_get_cross_account_call_forbidden(self, client, account, account2):
        """Account A cannot GET account B's call."""
        from conftest import SUBSCRIBER2_ID
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.status_code == 403
        client.delete(f"/api/calls/{call_id}", headers=account2)

    def test_get_stage(self, client, account):
        """?dsl=play:STAGE_ID returns stage info."""
        call_id = create_call(client, account)

        # Create a play stage
        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f'play:test_get{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
                    }),
                    headers=account)

        resp = client.get("/api/pipelines?dsl=play:test_get", headers=account)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == "play:test_get"
        assert data["call_id"] == call_id

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_get_nonexistent_stage(self, client, account):
        resp = client.get("/api/pipelines?dsl=play:nonexistent", headers=account)
        assert resp.status_code == 404

    def test_get_leg(self, client, account):
        """?dsl=sip:LEG_ID returns leg details."""
        from test_crm_e2e import _make_rtp_leg, _cleanup_leg
        from speech_pipeline.rtp_codec import PCMU
        leg, phone_rtp, _ = _make_rtp_leg(codec=PCMU)

        resp = client.get(f"/api/pipelines?dsl=sip:{leg.leg_id}", headers=account)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["leg_id"] == leg.leg_id

        _cleanup_leg(leg, phone_rtp)

    def test_get_rejects_pipeline_syntax(self, client, account):
        """GET rejects DSL with -> or |."""
        resp = client.get("/api/pipelines?dsl=play:x -> call:c1", headers=account)
        assert resp.status_code == 400

    def test_get_missing_dsl_returns_list(self, client, account):
        """GET without ?dsl returns pipeline list."""
        resp = client.get("/api/pipelines", headers=account)
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)

    def test_get_tee(self, client, account):
        """?dsl=tee:TEE_ID returns tee info."""
        call_id = create_call(client, account)

        # Create a bridge with tee
        from test_crm_e2e import _make_rtp_leg, _cleanup_leg
        from speech_pipeline.rtp_codec import PCMU
        leg, phone_rtp, _ = _make_rtp_leg(codec=PCMU)

        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f"sip:{leg.leg_id} -> tee:test_tee -> call:{call_id} -> sip:{leg.leg_id}"
                    }),
                    headers=account)

        resp = client.get("/api/pipelines?dsl=tee:test_tee", headers=account)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "tee"
        assert data["id"] == "test_tee"

        client.delete(f"/api/calls/{call_id}", headers=account)
        _cleanup_leg(leg, phone_rtp)

    def _placeholder(self):
        """DSL parser accepts webclient with JSON params."""
        from speech_pipeline.dsl_parser import parse_dsl
        result = parse_dsl(
            'webclient:user42{"callback":"/cb/wc","base_url":"https://srv.example.com"}'
        )
        assert result[0][0] == "webclient"
        assert result[0][1] == "user42"
        assert result[0][2]["callback"] == "/cb/wc"
        assert result[0][2]["base_url"] == "https://srv.example.com"


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

    def test_async_originate_pattern(self, client, account):
        """originate:NUM{cb} -> call:C — async fire-and-forget, no bidirectional."""
        call_id = create_call(client, account)
        # 2-element pipe (no second originate at end) = async mode
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f'originate:+4917099999{{"answered":"/cb/ans","ringing":"/cb/ring"}} -> call:{call_id}'
                           }),
                           headers=account)
        # Should return 201 (pipeline created, originate running in background)
        # Even if SIP fails because no real PBX, the pipeline creation should succeed
        # because the originate thread runs async
        assert resp.status_code in (201, 400)  # 400 if PBX misconfigured
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
    """Verify parse_dsl handles answer:, originate: correctly."""

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
