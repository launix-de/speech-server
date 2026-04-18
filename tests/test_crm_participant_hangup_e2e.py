"""End-to-end coverage for SIP participant hangup -> CRM cleanup -> call end.

These tests cover the production-relevant path that previously slipped
through green suites:

1. A SIP participant is bridged into a call via the real Pipe API.
2. The participant hangs up via the leg's real completion monitor.
3. The server fires the CRM ``state=leg&event=completed`` webhook.
4. The CRM marks the participant completed, runs ``checkLiveliness()``,
   and tears down the main call via
   ``DELETE /api/pipelines?dsl=call:<sid>``.
"""
from __future__ import annotations

import json
import time

import pytest

from conftest import ACCOUNT_ID, ACCOUNT_TOKEN, SUBSCRIBER_ID
from fake_crm import FakeCrm
from test_crm_e2e import _cleanup_leg, _make_rtp_leg
from speech_pipeline.telephony import call_state


@pytest.fixture
def crm(client, admin):
    acct = {
        "Authorization": f"Bearer {ACCOUNT_TOKEN}",
        "Content-Type": "application/json",
    }
    client.put(
        "/api/pbx/TestPBX",
        data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
        headers=admin,
    )
    client.put(
        f"/api/accounts/{ACCOUNT_ID}",
        data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX"}),
        headers=admin,
    )
    c = FakeCrm(client, admin_headers=admin, account_token=ACCOUNT_TOKEN)
    c.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")
    yield c


def _wait_for(predicate, *, timeout: float = 3.0, step: float = 0.05,
              failure: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(step)
    raise AssertionError(failure)


def _completed_events(crm: FakeCrm, *, participant_id: int) -> list[dict]:
    return [
        w for w in crm.webhooks
        if w["state"] == "leg"
        and w["query"].get("event") == "completed"
        and int(w["query"].get("participant", 0)) == participant_id
    ]


class TestParticipantHangupEndsMainCall:

    def test_bridged_sip_participant_hangup_ends_call(
            self, client, account, crm, monkeypatch):
        """A bridged SIP participant BYE must complete the CRM row and
        tear down the main call on the server."""
        leg, phone_rtp, _session = _make_rtp_leg(number="+491701111")
        call_sid = ""
        pid = 7001
        call_db_id = 701

        try:
            with crm.active(monkeypatch):
                resp = client.post(
                    "/api/calls",
                    data=json.dumps({
                        "subscriber_id": SUBSCRIBER_ID,
                        "caller": "+491701111",
                        "callee": "+4935000",
                        "direction": "outbound",
                    }),
                    headers=account,
                )
                assert resp.status_code == 201, resp.data
                call_sid = resp.get_json()["call_id"]

                crm.calls[call_db_id] = {
                    "caller": "+491701111",
                    "direction": "outbound",
                    "status": "answered",
                    "sid": call_sid,
                }
                crm.participants[pid] = {
                    "call_db_id": call_db_id,
                    "sid": leg.leg_id,
                    "status": "answered",
                    "number": "+491701111",
                }

                cb_completed = (
                    f"/Telephone/SpeechServer/public?call={call_db_id}"
                    f"&participant={pid}&state=leg&event=completed"
                )
                dsl = (
                    f"sip:{leg.leg_id}{json.dumps({'completed': cb_completed})}"
                    f" -> call:{call_sid} -> sip:{leg.leg_id}"
                )
                resp = client.post(
                    "/api/pipelines",
                    data=json.dumps({"dsl": dsl}),
                    headers=account,
                )
                assert resp.status_code == 201, resp.data

                time.sleep(0.2)
                leg.sip_session.hungup.set()

                _wait_for(
                    lambda: crm.participants[pid]["status"] == "completed",
                    failure=(
                        "CRM participant row never reached completed after "
                        "bridged SIP hangup"
                    ),
                )
                _wait_for(
                    lambda: crm.calls[call_db_id]["status"] == "completed",
                    failure=(
                        "CRM call row never completed after sole SIP "
                        "participant hangup"
                    ),
                )
                _wait_for(
                    lambda: call_state.get_call(call_sid) is None,
                    failure=(
                        "Server-side call still exists after CRM "
                        "checkLiveliness() should have deleted it"
                    ),
                )

                completed_events = _completed_events(crm, participant_id=pid)
                assert completed_events, (
                    "No participant-scoped completed webhook reached the CRM"
                )
        finally:
            _cleanup_leg(leg, phone_rtp)
            if call_sid and call_state.get_call(call_sid) is not None:
                client.delete(f"/api/pipelines?dsl=call:{call_sid}",
                              headers=account)
