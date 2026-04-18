"""Regression: call teardown must not synthesize a second incoming completed."""
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


class TestNoLateIncomingCompleted:

    def test_participant_hangup_does_not_trigger_second_call_level_completed(
            self, client, account, crm, monkeypatch):
        """After CRM-driven ``DELETE /api/pipelines?dsl=call:...``, the incoming caller leg
        must be torn down locally without emitting a second call-level
        ``state=leg&event=completed`` webhook."""
        incoming_leg, incoming_phone, _ = _make_rtp_leg(number="+491700001")
        participant_leg, participant_phone, _ = _make_rtp_leg(number="carli@test")
        call_sid = ""
        call_db_id = 0
        pid = 9190

        try:
            with crm.active(monkeypatch):
                resp = client.post(
                    "/api/calls",
                    data=json.dumps({
                        "subscriber_id": SUBSCRIBER_ID,
                        "caller": "+491700001",
                        "callee": "+4935000",
                        "direction": "inbound",
                    }),
                    headers=account,
                )
                assert resp.status_code == 201, resp.data
                call_sid = resp.get_json()["call_id"]
                call_db_id = 2701
                crm.calls[call_db_id] = {
                    "caller": "+491700001",
                    "direction": "inbound",
                    "status": "answered",
                    "sid": call_sid,
                }

                incoming_cb = (
                    f"/Telephone/SpeechServer/public?state=leg&event=completed"
                    f"&call={call_db_id}"
                )
                incoming_dsl = (
                    f"sip:{incoming_leg.leg_id}"
                    f"{json.dumps({'completed': incoming_cb})}"
                    f" -> call:{call_sid} -> sip:{incoming_leg.leg_id}"
                )
                resp = client.post(
                    "/api/pipelines",
                    data=json.dumps({"dsl": incoming_dsl}),
                    headers=account,
                )
                assert resp.status_code == 201, resp.data

                crm.participants[pid] = {
                    "call_db_id": call_db_id,
                    "sid": participant_leg.leg_id,
                    "status": "answered",
                    "number": "carli@test",
                }
                cb_completed = (
                    f"/Telephone/SpeechServer/public?call={call_db_id}"
                    f"&participant={pid}&state=leg&event=completed"
                )
                dsl = (
                    f"sip:{participant_leg.leg_id}"
                    f"{json.dumps({'completed': cb_completed})}"
                    f" -> call:{call_sid} -> sip:{participant_leg.leg_id}"
                )
                resp = client.post(
                    "/api/pipelines",
                    data=json.dumps({"dsl": dsl}),
                    headers=account,
                )
                assert resp.status_code == 201, resp.data

                baseline = len(crm.webhooks)
                participant_leg.sip_session.hungup.set()

                _wait_for(
                    lambda: call_state.get_call(call_sid) is None,
                    failure="CRM did not tear down the server-side call",
                )

                followup = crm.webhooks[baseline:]
                participant_completed = [
                    w for w in followup
                    if w["state"] == "leg"
                    and w["query"].get("event") == "completed"
                    and int(w["query"].get("participant", 0)) == pid
                ]
                call_level_completed = [
                    w for w in followup
                    if w["state"] == "leg"
                    and w["query"].get("event") == "completed"
                    and int(w["query"].get("call", 0)) == call_db_id
                    and "participant" not in w["query"]
                ]

                assert participant_completed, (
                    "participant completed webhook never reached the CRM"
                )
                assert not call_level_completed, (
                    "incoming leg emitted a late call-level completed webhook "
                    "during local call teardown"
                )
        finally:
            _cleanup_leg(incoming_leg, incoming_phone)
            _cleanup_leg(participant_leg, participant_phone)
            if call_sid and call_state.get_call(call_sid) is not None:
                client.delete(f"/api/pipelines?dsl=call:{call_sid}",
                              headers=account)
