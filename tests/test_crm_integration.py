"""CRM integration surface tests.

Locks in the exact request sequences used by the CRM (fop-dev,
``backends/businesslogic/telefonanlage/speech-server``) so we can't
silently break the deployed integration.

Each ``TestCrmFlow*`` class mirrors a specific CRM code path by
reproducing the same HTTP calls the ``.fop`` files make.  If a CRM
contract changes, these tests must be updated in lockstep.
"""
from __future__ import annotations

import json
import time
from typing import Tuple
from unittest.mock import patch, MagicMock

import pytest

from conftest import SUBSCRIBER_ID, create_call
from speech_pipeline.telephony import call_state


# ---------------------------------------------------------------------------
# CRM: settings.fop  — stopWaitMusic / stopHoldMusic
# ---------------------------------------------------------------------------

class TestCrmStopMusic:
    """``stopWaitMusic(callSid)`` / ``stopHoldMusic(callSid, legSid)``
    issue ``DELETE /api/pipelines`` with the plain DSL item (no ``kill:``
    prefix)."""

    def test_stop_wait_music_via_dsl_delete(self, client, account):
        call_id = create_call(client, account)
        try:
            wait_stage_id = f"play:{call_id}_wait"
            # Start the wait music (matches outgoing.fop wording).
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f'play:{call_id}_wait{{"url":"examples/queue.mp3",'
                           f'"loop":true,"volume":50}} -> call:{call_id}'
                }),
                headers=account,
            )
            assert resp.status_code == 201

            # Stage is now findable via GET ?dsl=play:ID.
            resp = client.get(
                f"/api/pipelines?dsl={wait_stage_id}", headers=account
            )
            assert resp.status_code == 200

            # CRM's stopWaitMusic: DELETE with the same plain item.
            resp = client.delete(
                "/api/pipelines",
                data=json.dumps({"dsl": wait_stage_id}),
                headers=account,
            )
            assert resp.status_code == 204

            # After delete, stage is gone.
            resp = client.get(
                f"/api/pipelines?dsl={wait_stage_id}", headers=account
            )
            assert resp.status_code == 404
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_stop_hold_music_via_dsl_delete(self, client, account):
        call_id = create_call(client, account)
        leg_sid = "leg-fake-hold-1"
        try:
            # CRM hold flow writes `play:{call}_hold_{leg}`.
            hold_stage_id = f"play:{call_id}_hold_{leg_sid}"
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f'play:{call_id}_hold_{leg_sid}'
                           f'{{"url":"examples/queue.mp3","loop":true}} '
                           f'-> call:{call_id}'
                }),
                headers=account,
            )
            assert resp.status_code == 201

            # stopHoldMusic — CRM fires DELETE with the plain item.
            resp = client.delete(
                "/api/pipelines",
                data=json.dumps({"dsl": hold_stage_id}),
                headers=account,
            )
            assert resp.status_code == 204
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_legacy_kill_prefix_rejected(self, client, account):
        """CRM must NOT send ``POST /api/pipelines {dsl: kill:...}`` any more."""
        call_id = create_call(client, account)
        try:
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": f"kill:play:{call_id}_wait"}),
                headers=account,
            )
            # The parser accepts `kill` as a bare element name, but the
            # executor has no creator for it → either validation error
            # (no neighbour) or stage creation error.  Either way: 4xx.
            assert 400 <= resp.status_code < 500
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# CRM: calls.fop  — findExternalLegIds
# ---------------------------------------------------------------------------

class TestCrmFindExternalLegs:
    """``findExternalLegIds(id)`` reads ``call.participants[]`` via the
    new DSL-based lookup."""

    def test_call_lookup_returns_participants_field(self, client, account):
        call_id = create_call(client, account)
        try:
            resp = client.get(
                f"/api/pipelines?dsl=call:{call_id}", headers=account
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "participants" in data
            assert isinstance(data["participants"], list)
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_call_lookup_includes_participant_metadata(self, client, account):
        """After a participant is registered, the lookup surfaces it
        with id/type/direction — the fields CRM's findExternalLegIds
        filters on (``type == 'sip' && direction == 'inbound'``)."""
        call_id = create_call(client, account)
        try:
            call = call_state.get_call(call_id)
            call.register_participant(
                "leg-fake-123",
                type="sip",
                direction="inbound",
                number="+49123",
            )
            resp = client.get(
                f"/api/pipelines?dsl=call:{call_id}", headers=account
            )
            assert resp.status_code == 200
            parts = resp.get_json()["participants"]
            assert len(parts) == 1
            p = parts[0]
            assert p["id"] == "leg-fake-123"
            assert p["type"] == "sip"
            assert p["direction"] == "inbound"
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# CRM: webclient.fop  — join via DSL
# ---------------------------------------------------------------------------

class TestCrmWebclientJoin:
    """``joinWebclientAction`` posts
    ``POST /api/pipelines {dsl: "webclient:USER{callback,base_url,call_id}"}``.
    The call_id param binds the executor so the action can run, but the
    CRM callback itself must wait for the browser's real ``answered``
    event instead of firing during slot creation."""

    def test_webclient_with_call_id_creates_session(self, client, account):
        """The DSL action must resolve the call context via ``call_id``
        param (no ``call:`` element in the DSL).  This mirrors exactly
        what ``webclient.fop::joinWebclientAction`` sends."""
        call_id = create_call(client, account)
        try:
            with patch(
                "speech_pipeline.telephony._shared.post_webhook",
            ) as mock_post:
                params = json.dumps({
                    "callback": "/cb/webclient",
                    "base_url": "https://crm.example.com",
                    "call_id": call_id,
                })
                resp = client.post(
                    "/api/pipelines",
                    data=json.dumps({"dsl": f"webclient:user1{params}"}),
                    headers=account,
                )

                if resp.status_code == 400 and b"webclient feature" in resp.data:
                    pytest.skip(
                        "Account lacks webclient feature — CRM integration "
                        "needs this enabled in production."
                    )
                assert resp.status_code == 201
                body = resp.get_json()
                assert body.get("session_id", "").startswith("wc-")
                assert body.get("nonce", "").startswith("n-")
                assert body.get("iframe_url", "").startswith(
                    "https://crm.example.com/phone/"
                )

                # Slot creation alone must NOT tell the CRM the browser
                # joined yet.
                assert not mock_post.called

                resp2 = client.post(
                    f"/phone/{body['nonce']}/event",
                    data=json.dumps({
                        "session": body["session_id"],
                        "event": "answered",
                    }),
                    headers={"Content-Type": "application/json"},
                )
                assert resp2.status_code == 200

            # Participant registered as webclient on the call.
            lookup = client.get(
                f"/api/pipelines?dsl=call:{call_id}", headers=account,
            ).get_json()
            wc_parts = [p for p in lookup["participants"]
                        if p.get("type") == "webclient"]
            assert len(wc_parts) == 1
            # The callback webhook now fires on real browser join.
            assert mock_post.called
            _args, _kwargs = mock_post.call_args
            payload = _args[1] if len(_args) > 1 else _kwargs.get("payload", {})
            assert "iframe_url" in payload
            assert payload["iframe_url"].startswith("https://crm.example.com/phone/")
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_webclient_without_call_id_fails(self, client, account):
        """No call_id, no call: element → no context, must fail clearly."""
        resp = client.post(
            "/api/pipelines",
            data=json.dumps({
                "dsl": 'webclient:u1{"callback":"/cb","base_url":"https://x"}'
            }),
            headers=account,
        )
        assert resp.status_code == 400

    def test_webclient_with_wrong_call_id_fails(self, client, account):
        resp = client.post(
            "/api/pipelines",
            data=json.dumps({
                "dsl": 'webclient:u1{"callback":"/cb",'
                       '"base_url":"https://x","call_id":"call-bogus"}'
            }),
            headers=account,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Legacy endpoints are really gone
# ---------------------------------------------------------------------------

class TestCrmOriginate:
    """``calls.fop::findParticipant`` posts
    ``originate:NUMBER{cb} -> call:SID`` and reads ``leg_id`` from the
    response.  The leg_id gets stored as the participant SID for
    callback routing.  Regression guard for the `/api/legs/originate`
    404 that broke outgoing calls in production."""

    def test_originate_response_contains_leg_id(self, client, account, monkeypatch):
        """POST /api/pipelines with `originate:NUM -> call:C` returns leg_id."""
        from speech_pipeline.telephony import pbx as pbx_mod
        from speech_pipeline.telephony import leg as leg_mod

        # Register a PBX so originate can resolve it.  Default-test PBX
        # already exists from conftest, but be explicit.
        pbx_mod.put("TestPBX", {
            "id": "TestPBX",
            "sip_server": "sip.example.com",
            "sip_port": 5060,
            "realm": "sip.example.com",
        })
        # Stop originate_only from actually dialing.
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)

        # Create call bound to this pbx.
        resp = client.post(
            "/api/calls",
            data=json.dumps({
                "subscriber_id": SUBSCRIBER_ID,
                "pbx": "TestPBX",
            }),
            headers=account,
        )
        assert resp.status_code == 201, resp.data
        call_id = resp.get_json()["call_id"]

        try:
            cb = {
                "ringing": "/cb/ring",
                "answered": "/cb/ans",
                "completed": "/cb/done",
                "failed": "/cb/fail",
                "no-answer": "/cb/na",
                "busy": "/cb/busy",
            }
            dsl = f"originate:+491234567890{json.dumps(cb)} -> call:{call_id}"
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": dsl}),
                headers=account,
            )
            assert resp.status_code == 201, resp.data
            body = resp.get_json()
            assert "leg_id" in body, f"leg_id missing from response: {body}"
            assert body["leg_id"].startswith("leg-")
            # Leg is registered on the server side.
            assert leg_mod.get_leg(body["leg_id"]) is not None
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_originate_to_unknown_call_rejected(self, client, account):
        """``-> call:nonexistent`` → 4xx (not 404 from a missing route)."""
        dsl = ('originate:+491234567890{"ringing":"/cb"} '
               '-> call:call-does-not-exist')
        resp = client.post(
            "/api/pipelines",
            data=json.dumps({"dsl": dsl}),
            headers=account,
        )
        # Account not allowed to post to a call it does not own → 403,
        # or "call not found" → 400.  Both acceptable; 404 (route-level)
        # would mean the whole endpoint vanished and is not acceptable.
        assert 400 <= resp.status_code < 500
        assert resp.status_code != 404


class TestLegacyEndpointsRemoved:
    """Guardrail: these endpoints were removed in favour of DSL.  If a
    future refactor re-introduces them we fail loudly."""

    def test_get_call_detail_endpoint_removed(self, client, account):
        call_id = create_call(client, account)
        try:
            resp = client.get(f"/api/calls/{call_id}", headers=account)
            assert resp.status_code == 405  # method not allowed on /<id>
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_participants_endpoint_removed(self, client, account):
        call_id = create_call(client, account)
        try:
            resp = client.get(
                f"/api/calls/{call_id}/participants", headers=account
            )
            assert resp.status_code == 404
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_legs_originate_endpoint_removed(self, client, account):
        """The old ``POST /api/legs/originate`` endpoint must return
        404 — replaced by ``POST /api/pipelines {dsl: "originate:..."}``."""
        resp = client.post("/api/legs/originate", headers=account,
                           data=json.dumps({"call_id": "x", "to": "+1"}))
        assert resp.status_code == 404

    def test_commands_endpoint_removed(self, client, account):
        call_id = create_call(client, account)
        try:
            resp = client.post(
                f"/api/calls/{call_id}/commands",
                data=json.dumps({"commands": [{"action": "play"}]}),
                headers=account,
            )
            assert resp.status_code == 404
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_commands_module_gone(self):
        with pytest.raises(ImportError):
            from speech_pipeline.telephony import commands  # noqa: F401

    def test_call_has_no_command_queue(self, client, account):
        call_id = create_call(client, account)
        try:
            call = call_state.get_call(call_id)
            assert not hasattr(call, "command_queue")
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# CRM: outgoing.fop / webhooks.fop — DSL wording still parses
# ---------------------------------------------------------------------------

class TestCrmDslStringsParse:
    """The exact DSL strings the CRM constructs (with JSON params,
    callbacks, wait/hold stage names) must parse — this is a
    syntactic guardrail independent of execution."""

    @pytest.mark.parametrize("dsl", [
        # outgoing.fop:30 wait music + source into conference
        'play:call-X_wait{"url":"https://crm/jingle","loop":true,'
        '"volume":60} -> call:call-X',
        # outgoing.fop:50 / webhooks.fop:67  bidirectional bridge
        'sip:legY{"completed":"/cb/done"} -> call:call-X -> sip:legY',
        # outgoing.fop:66  async originate
        'originate:+491234567{"ringing":"/cb/ring","answered":"/cb/ans",'
        '"completed":"/cb/done","failed":"/cb/fail","no-answer":"/cb/na",'
        '"busy":"/cb/busy","canceled":"/cb/cancel",'
        '"caller_id":"Anruf CRM"} -> call:call-X',
        # webhooks.fop:58  TTS denial message
        'tts:de{"text":"Geschlossen","completed":"/cb/tts"} -> call:call-X',
        # webclient.fop  webclient join (new)
        'webclient:user_42{"callback":"/cb/wc",'
        '"base_url":"https://crm.example.com","call_id":"call-X"}',
        # calls.fop:149  hold music targeting a specific leg
        'play:call-X_hold_legY{"url":"https://crm/hold","loop":true,'
        '"volume":50} -> sip:legY',
    ])
    def test_parses(self, dsl):
        from speech_pipeline.dsl_parser import parse_dsl
        result = parse_dsl(dsl)
        assert result, f"empty parse for: {dsl!r}"
