"""Regression tests for inbound trunk-leg cleanup through call teardown.

These tests cover the production path that was still broken:

1. a trunk inbound SIP leg is coupled into a conference as a ConferenceLeg
2. the call is ended via call teardown
3. the coupled leg must be removed from the server registry
4. the conference source/sink wiring must be detached
5. the SIP cleanup must send real signaling to the trunk side

The last point is the critical one: once the inbound leg has been
answered, call teardown must emit a SIP BYE to the trunk peer.
"""
from __future__ import annotations

import json
import time

from conftest import create_call
from speech_pipeline.telephony import call_state, leg as leg_mod, sip_stack
from test_crm_e2e import _cleanup_leg, _make_rtp_leg


def _install_trunk_dialog(leg, *, answered: bool = False) -> str:
    """Attach a fake trunk-dialog state to an RTP-backed inbound leg."""
    sip_call_id = f"dlg-{leg.leg_id}"
    leg.sip_call_id = sip_call_id
    sip_stack._trunk_dialogs[sip_call_id] = {
        "msg": {
            "headers": {
                "to": "<sip:4935863190000@test.local>",
                "from": "<sip:+491747712705@trunk.example>;tag=fromtag",
                "via": (
                    "SIP/2.0/UDP trunk.example:5060;"
                    "branch=z9hG4bK-test-branch"
                ),
                "call-id": sip_call_id,
                "cseq": "1 INVITE",
                "contact": "<sip:+491747712705@trunk.example:5060>",
            },
            "body": "",
        },
        "addr": ("127.0.0.1", 5060),
        "to_tag": "localtotag",
        "sdp": "v=0\r\n",
        "answered": answered,
        "leg_id": leg.leg_id,
    }
    return sip_call_id


class TestInboundTrunkCleanup:
    def test_delete_call_cleans_up_coupled_leg_registry(
            self, client, account):
        call_id = create_call(client, account)
        leg, phone_rtp, _session = _make_rtp_leg(number="+491747712705")
        sip_call_id = _install_trunk_dialog(leg)

        try:
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                }),
                headers=account,
            )
            assert resp.status_code == 201, resp.data
            assert leg_mod.get_leg(leg.leg_id) is not None

            resp = client.delete(
                f"/api/pipelines?dsl=call:{call_id}",
                headers=account,
            )
            assert resp.status_code == 204
            assert call_state.get_call(call_id) is None
            assert leg_mod.get_leg(leg.leg_id) is None
        finally:
            sip_stack._trunk_dialogs.pop(sip_call_id, None)
            _cleanup_leg(leg, phone_rtp)

    def test_delete_call_detaches_coupled_source_and_sink(
            self, client, account):
        call_id = create_call(client, account)
        leg, phone_rtp, _session = _make_rtp_leg(number="+491747712705")
        sip_call_id = _install_trunk_dialog(leg)

        try:
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                }),
                headers=account,
            )
            assert resp.status_code == 201, resp.data

            call = call_state.get_call(call_id)
            conf_leg = call.pipe_executor._stages[f"bridge:{leg.leg_id}"]._stage
            deadline = time.time() + 1.0
            while ((getattr(conf_leg, "_src_id", None) is None
                    or getattr(conf_leg, "_sink_id", None) is None)
                   and time.time() < deadline):
                time.sleep(0.02)

            src_id = conf_leg._src_id
            sink_id = conf_leg._sink_id
            assert src_id is not None
            assert sink_id is not None
            assert src_id in call.mixer._sources
            assert any(entry.id == sink_id for entry in call.mixer._sinks)

            resp = client.delete(
                f"/api/pipelines?dsl=call:{call_id}",
                headers=account,
            )
            assert resp.status_code == 204
            assert src_id not in call.mixer._sources
            deadline = time.time() + 1.0
            while any(entry.id == sink_id for entry in call.mixer._sinks) and time.time() < deadline:
                time.sleep(0.02)
            assert all(entry.id != sink_id for entry in call.mixer._sinks)
        finally:
            sip_stack._trunk_dialogs.pop(sip_call_id, None)
            _cleanup_leg(leg, phone_rtp)

    def test_answered_trunk_leg_cleanup_sends_bye(
            self, client, account, monkeypatch):
        call_id = create_call(client, account)
        leg, phone_rtp, _session = _make_rtp_leg(number="+491747712705")
        sip_call_id = _install_trunk_dialog(leg, answered=False)
        sent_messages: list[str] = []

        def _capture_send(msg: str, addr):
            sent_messages.append(msg)

        monkeypatch.setattr(sip_stack, "_send", _capture_send)

        try:
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
                }),
                headers=account,
            )
            assert resp.status_code == 201, resp.data

            resp = client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": f"answer:{leg.leg_id}"}),
                headers=account,
            )
            assert resp.status_code == 201
            assert sip_stack._trunk_dialogs[sip_call_id]["answered"] is True
            assert any("SIP/2.0 200 OK" in msg for msg in sent_messages), (
                "answer:LEG did not send 200 OK for the inbound trunk leg"
            )

            resp = client.delete(
                f"/api/pipelines?dsl=call:{call_id}",
                headers=account,
            )
            assert resp.status_code == 204
            assert any(msg.startswith("BYE ") for msg in sent_messages), (
                "cleanup of the answered inbound trunk leg did not send BYE"
            )
            assert not any(msg.startswith("CANCEL ") for msg in sent_messages), (
                "answered trunk-leg cleanup must send BYE, not CANCEL"
            )
        finally:
            sip_stack._trunk_dialogs.pop(sip_call_id, None)
            _cleanup_leg(leg, phone_rtp)
