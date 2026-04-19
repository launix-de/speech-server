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
from test_crm_e2e import SUBSCRIBER_ID, _cleanup_leg, _make_rtp_leg


ACCOUNT_SCOPE = "test-account"


def _scope_leg_for_account(leg) -> str:
    local_leg_id = leg.leg_id
    scoped_leg_id = f"{ACCOUNT_SCOPE}:{local_leg_id}"
    leg_mod._legs.pop(local_leg_id, None)
    leg.leg_id = scoped_leg_id
    leg.subscriber_id = SUBSCRIBER_ID
    leg_mod._legs[scoped_leg_id] = leg
    return local_leg_id


def _scoped_call_id(local_call_id: str) -> str:
    return f"{ACCOUNT_SCOPE}:{local_call_id}"


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
        local_leg_id = _scope_leg_for_account(leg)
        sip_call_id = _install_trunk_dialog(leg)

        try:
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{local_leg_id} -> call:{call_id} -> sip:{local_leg_id}"
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
        local_leg_id = _scope_leg_for_account(leg)
        sip_call_id = _install_trunk_dialog(leg)

        try:
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{local_leg_id} -> call:{call_id} -> sip:{local_leg_id}"
                }),
                headers=account,
            )
            assert resp.status_code == 201, resp.data

            call = call_state.get_call(_scoped_call_id(call_id))
            conf_leg = call.pipe_executor._stages[f"bridge:{local_leg_id}"]._stage
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
        local_leg_id = _scope_leg_for_account(leg)
        sip_call_id = _install_trunk_dialog(leg, answered=False)
        sent_messages: list[str] = []

        def _capture_send(msg: str, addr):
            sent_messages.append(msg)

        monkeypatch.setattr(sip_stack, "_send", _capture_send)

        try:
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{local_leg_id} -> call:{call_id} -> sip:{local_leg_id}"
                }),
                headers=account,
            )
            assert resp.status_code == 201, resp.data

            resp = client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": f"answer:{local_leg_id}"}),
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

    def test_remote_end_of_inbound_rtp_leg_fires_completed_callback(
            self, client, account, monkeypatch):
        call_id = create_call(client, account)
        leg, phone_rtp, session = _make_rtp_leg(number="+491747712705")
        local_leg_id = _scope_leg_for_account(leg)
        sip_call_id = _install_trunk_dialog(leg, answered=True)
        callbacks: list[tuple[str, dict]] = []

        def _capture_callback(_url, payload, _token, **_kw):
            callbacks.append((_url, dict(payload or {})))

        monkeypatch.setattr("speech_pipeline.telephony._shared.post_webhook", _capture_callback)

        try:
            leg.callbacks["completed"] = "/cb?call=1&participant=0&state=leg&event=completed"

            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{local_leg_id} -> call:{call_id} -> sip:{local_leg_id}"
                }),
                headers=account,
            )
            assert resp.status_code == 201, resp.data
            deadline = time.time() + 1.0
            while time.time() < deadline and not getattr(leg, "completion_monitor_started", False):
                time.sleep(0.02)
            assert getattr(leg, "completion_monitor_started", False), (
                "bridged inbound leg did not start its completion monitor"
            )

            # Simulate the production failure mode: RTP input dries up and the
            # source reaches natural EOF without an explicit local delete_call.
            session.rx_queue.put(b"\x00\x00" * 160)
            session.rx_queue.put(None)

            deadline = time.time() + 3.0
            while time.time() < deadline and not callbacks:
                time.sleep(0.05)

            assert callbacks, (
                "natural EOF of an inbound RTP/trunk leg did not fire the "
                "completed webhook; CRM never learns that the external caller "
                "hung up"
            )
            assert callbacks[0][1]["event"] == "completed"
            assert callbacks[0][1]["leg_id"] == local_leg_id
        finally:
            sip_stack._trunk_dialogs.pop(sip_call_id, None)
            _cleanup_leg(leg, phone_rtp)
