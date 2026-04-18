"""E2E: internal SIP participant hangup on device_dial must end the call."""
from __future__ import annotations

import json
import time

import pytest

from conftest import ACCOUNT_ID, ACCOUNT_TOKEN, SUBSCRIBER_ID
from fake_crm import FakeCrm
from test_crm_e2e import _cleanup_leg, _make_rtp_leg
from speech_pipeline.telephony import (
    call_state,
    dispatcher,
    leg as leg_mod,
    sip_stack,
    subscriber as sub_mod,
)


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


class TestDeviceDialParticipantHangup:

    def test_internal_sip_participant_hangup_ends_main_call(
            self, client, account, crm, monkeypatch):
        """The CRM tracks the dialing SIP client as the only participant.
        When that client hangs up, ``checkLiveliness()`` must end the main
        call and the server must delete the conference."""
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)

        internal_leg, phone_rtp, _session = _make_rtp_leg(number="admin")
        call_sid = ""

        try:
            with crm.active(monkeypatch):
                sub = sub_mod.get(SUBSCRIBER_ID)
                dispatcher.fire_subscriber_event(sub, "device_dial", {
                    "number": "+4935863",
                    "leg_id": internal_leg.leg_id,
                    "sip_user": "admin",
                    "user_id": 42,
                })

                _wait_for(
                    lambda: len(crm.calls) == 1 and len(crm.participants) == 1,
                    failure=(
                        "device_dial did not create exactly one call row and "
                        "one internal participant row in the CRM"
                    ),
                )

                call_db_id = next(iter(crm.calls.keys()))
                call_sid = crm.calls[call_db_id]["sid"]
                pid = next(iter(crm.participants.keys()))

                assert crm.participants[pid]["sid"] == internal_leg.leg_id
                assert crm.participants[pid]["status"] == "answered"

                internal_leg.sip_session.hungup.set()

                _wait_for(
                    lambda: crm.participants[pid]["status"] == "completed",
                    failure=(
                        "CRM did not mark the internal SIP participant "
                        "completed after hangup"
                    ),
                )
                _wait_for(
                    lambda: crm.calls[call_db_id]["status"] == "completed",
                    failure=(
                        "CRM did not complete the main call after the sole "
                        "internal SIP participant hung up"
                    ),
                )
                _wait_for(
                    lambda: call_state.get_call(call_sid) is None,
                    failure=(
                        "Server-side call still existed after the CRM "
                        "completed the device_dial call"
                    ),
                )
        finally:
            _cleanup_leg(internal_leg, phone_rtp)
            if call_sid and call_state.get_call(call_sid) is not None:
                client.delete(f"/api/pipelines?dsl=call:{call_sid}",
                              headers=account)

    def test_internal_sip_bye_path_ends_main_call(
            self, client, account, crm, monkeypatch):
        """Use the real ``sip_stack._handle_inbound_bye`` path instead of
        manually setting ``session.hungup`` so the regression covers the
        production signaling hook."""
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)
        monkeypatch.setattr(sip_stack, "_send", lambda msg, addr: None)

        internal_leg, phone_rtp, session = _make_rtp_leg(number="admin")
        call_sid = ""
        sip_call_id = "dlg-device-dial-test"

        try:
            with crm.active(monkeypatch):
                sub = sub_mod.get(SUBSCRIBER_ID)
                dispatcher.fire_subscriber_event(sub, "device_dial", {
                    "number": "+4935863",
                    "leg_id": internal_leg.leg_id,
                    "sip_user": "admin",
                    "user_id": 42,
                })

                _wait_for(
                    lambda: len(crm.calls) == 1 and len(crm.participants) == 1,
                    failure="device_dial bootstrap did not reach CRM",
                )

                call_db_id = next(iter(crm.calls.keys()))
                call_sid = crm.calls[call_db_id]["sid"]
                pid = next(iter(crm.participants.keys()))

                internal_leg.sip_call_id = sip_call_id
                sip_stack._trunk_dialogs[sip_call_id] = {
                    "session": session,
                    "rtp_session": internal_leg.rtp_session,
                    "leg_id": internal_leg.leg_id,
                }

                sip_stack._handle_inbound_bye({
                    "headers": {
                        "call-id": sip_call_id,
                        "via": "SIP/2.0/UDP 127.0.0.1:5060;branch=z9hG4bK-test",
                        "from": "<sip:admin@crm.example>;tag=fromtag",
                        "to": "<sip:+4935863@test.local>;tag=totag",
                        "cseq": "2 BYE",
                    }
                }, ("127.0.0.1", 5060))

                _wait_for(
                    lambda: crm.participants[pid]["status"] == "completed",
                    failure=(
                        "CRM did not mark the internal SIP participant "
                        "completed after a real SIP BYE"
                    ),
                )
                _wait_for(
                    lambda: crm.calls[call_db_id]["status"] == "completed",
                    failure=(
                        "CRM did not complete the main call after the "
                        "internal SIP BYE path fired"
                    ),
                )
                _wait_for(
                    lambda: call_state.get_call(call_sid) is None,
                    failure=(
                        "Server-side call still existed after the BYE path "
                        "should have deleted it"
                    ),
                )
        finally:
            sip_stack._trunk_dialogs.pop(sip_call_id, None)
            _cleanup_leg(internal_leg, phone_rtp)
            if call_sid and call_state.get_call(call_sid) is not None:
                client.delete(f"/api/pipelines?dsl=call:{call_sid}",
                              headers=account)
