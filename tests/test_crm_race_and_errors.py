"""CRM race conditions + error-path coverage.

Gaps #6–10 from the audit:

6. ``_on_leg_answered`` when the call was already torn down.
7. Multi-participant teardown order: 3+ legs, hold, completed out of order.
8. Webclient dedup: second Join on same (call, user) must reuse.
9. Server returns non-201 on ``POST /api/pipelines`` — CRM must recover.
10. Account without ``webclient`` feature → action must fail loudly.
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from conftest import (
    ADMIN_TOKEN, ACCOUNT_TOKEN, ACCOUNT_ID, SUBSCRIBER_ID,
)
from fake_crm import FakeCrm
from speech_pipeline.telephony import (
    call_state, dispatcher, leg as leg_mod, subscriber as sub_mod,
)


@pytest.fixture
def crm(client, admin):
    acct = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
            "Content-Type": "application/json"}
    client.put("/api/pbx/TestPBX",
               data=json.dumps({"sip_proxy": "", "sip_user": "",
                                "sip_password": ""}), headers=admin)
    client.put(f"/api/accounts/{ACCOUNT_ID}",
               data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX",
                                 "features": ["webclient"]}),
               headers=admin)
    c = FakeCrm(client, admin_headers=admin, account_token=ACCOUNT_TOKEN)
    c.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")
    yield c


# ---------------------------------------------------------------------------
# #6. answered-after-hangup race
# ---------------------------------------------------------------------------

class TestAnsweredAfterCallEnded:
    """CRM receives state=leg&event=answered but the call was already
    torn down on its side.  Must not crash or create orphan pipes."""

    def test_answered_on_unknown_call_is_no_op(
            self, client, account, crm, monkeypatch):
        with crm.active(monkeypatch):
            crm._route(
                crm.BASE_URL + "/Telephone/SpeechServer/public"
                "?state=leg&event=answered&call=999&participant=99",
                {"leg_id": "leg-nowhere"},
                crm.account_token, method="POST",
            )
        # Must neither crash nor mutate random state.
        assert 999 not in crm.calls
        assert 99 not in crm.participants

    def test_answered_on_completed_call_does_not_rebuild_bridge(
            self, client, account, crm, monkeypatch):
        """Late answered-webhook after call went completed must not
        issue /api/pipelines calls (would leak stages)."""
        with crm.active(monkeypatch):
            call_db_id = 1
            crm.calls[call_db_id] = {
                "caller": "+49170", "direction": "inbound",
                "status": "completed", "sid": "",
            }
            before = len(crm.webhooks)
            crm._route(
                crm.BASE_URL + "/Telephone/SpeechServer/public"
                "?state=leg&event=answered&call=1&participant=1",
                {"leg_id": "leg-x"},
                crm.account_token, method="POST",
            )
            # Call stays completed, no new POSTs issued (other than the
            # replay entry itself in the audit trail).
            assert crm.calls[call_db_id]["status"] == "completed"
            assert len(crm.webhooks) == before + 1


# ---------------------------------------------------------------------------
# #7. Multi-participant teardown order
# ---------------------------------------------------------------------------

class TestMultiParticipantTeardownOrder:
    """3 legs: A answered, B on hold, C answered.  C ends, then B ends
    (while still on hold), A is last — call must end cleanly regardless
    of the order."""

    def test_three_way_teardown_does_not_leave_call_hanging(
            self, client, account, crm, monkeypatch):
        with crm.active(monkeypatch):
            resp = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                               headers=account)
            call_sid = resp.get_json()["call_id"]
            call_db_id = 7
            crm.calls[call_db_id] = {
                "caller": "", "direction": "outbound",
                "status": "answered", "sid": call_sid,
            }
            pids = {"A": 701, "B": 702, "C": 703}
            for name, pid in pids.items():
                crm.participants[pid] = {
                    "call_db_id": call_db_id,
                    "status": "hold" if name == "B" else "answered",
                    "number": f"+4900{pid}", "sid": f"leg-{name.lower()}",
                }

            def _cb(pid):
                return (f"/Telephone/SpeechServer/public?call={call_db_id}"
                        f"&participant={pid}")

            def _leg(pid):
                fake_voip = MagicMock(); fake_voip.RTPClients = []
                leg = leg_mod.create_leg(
                    direction="outbound", number="+x", pbx_id="TestPBX",
                    subscriber_id=SUBSCRIBER_ID, voip_call=fake_voip,
                )
                leg.call_id = call_sid
                leg.callbacks = {"completed": _cb(pid)
                                 + "&state=leg&event=completed"}
                return leg

            # C completes first.
            leg_c = _leg(pids["C"])
            leg_mod.fire_callback(leg_c, "completed", duration=3.0)
            # B (on hold) completes next — should not crash.
            leg_b = _leg(pids["B"])
            leg_mod.fire_callback(leg_b, "completed", duration=2.0)
            # A last — call must end.
            leg_a = _leg(pids["A"])
            leg_mod.fire_callback(leg_a, "completed", duration=6.0)
            time.sleep(0.3)

            assert crm.calls[call_db_id]["status"] == "completed", (
                f"Call never ended after all participants left: "
                f"{crm.calls[call_db_id]}"
            )
            for pid in pids.values():
                assert crm.participants[pid]["status"] == "completed", (
                    f"participant {pid} not marked completed: "
                    f"{crm.participants[pid]}"
                )
            for leg in (leg_a, leg_b, leg_c):
                leg_mod._legs.pop(leg.leg_id, None)


# ---------------------------------------------------------------------------
# #8. Webclient dedup: second webclient: call for same (call, user)
# ---------------------------------------------------------------------------

class TestWebclientDedup:
    """Current behaviour: server ``register_webclient`` creates a new
    session each time. CRM's ``joinWebclientAction`` guards against
    duplicate Participant rows, but the server still accepts every
    call. Until dedup is added server-side, two sessions will exist
    for the same (call, user).

    This test pins today's behaviour so a future dedup change is a
    conscious decision (the test has to be updated)."""

    def test_two_webclient_actions_produce_two_distinct_sessions(
            self, client, account, crm, monkeypatch):
        from speech_pipeline.telephony import webclient as wc_mod
        with crm.active(monkeypatch):
            resp = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                               headers=account)
            call_sid = resp.get_json()["call_id"]
            dsl_params = {
                "callback": "/cb?call=1&participant=1",
                "base_url": "https://crm.example.com",
                "call_id": call_sid,
            }
            s1 = client.post("/api/pipelines",
                             data=json.dumps({"dsl":
                                 'webclient:userA' + json.dumps(dsl_params)}),
                             headers=account).get_json()
            s2 = client.post("/api/pipelines",
                             data=json.dumps({"dsl":
                                 'webclient:userA' + json.dumps(dsl_params)}),
                             headers=account).get_json()
            assert s1["session_id"] != s2["session_id"], (
                "Server dedup landed without a matching test update — "
                "adjust this expectation to match the new contract."
            )
            client.delete(f"/api/calls/{call_sid}", headers=account)


# ---------------------------------------------------------------------------
# #9. Server returns non-201 on webclient POST
# ---------------------------------------------------------------------------

class TestWebclientServerErrorPath:
    """If the server rejects the webclient: action (bad DSL, missing
    feature, url_safety), the response must carry a non-201 status
    with a machine-readable error — so the CRM can mark the
    Participant as 'failed' instead of silently pretending to have
    opened an iframe."""

    def test_missing_base_url_returns_400(self, client, account):
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl":
                               'webclient:userA{"call_id":"bogus"}'}),
                           headers=account)
        assert resp.status_code in (400, 404), resp.get_data(as_text=True)

    def test_invalid_call_id_returns_error(self, client, account):
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl":
                               'webclient:userA{"callback":"/cb",'
                               '"base_url":"https://crm.example.com",'
                               '"call_id":"does-not-exist"}'}),
                           headers=account)
        assert resp.status_code != 201, (
            f"server silently accepted webclient: on unknown call: "
            f"{resp.status_code} {resp.get_data(as_text=True)}"
        )


# ---------------------------------------------------------------------------
# #10. Webclient feature disabled
# ---------------------------------------------------------------------------

class TestWebclientFeatureGate:
    """An account without the ``webclient`` feature must get a crisp
    400/403 on the DSL action — not a 500 or silent success."""

    def test_feature_disabled_rejects_action(self, client, admin):
        # Separate account with NO webclient feature.
        acct = {"Authorization": "Bearer no-wc-tok",
                "Content-Type": "application/json"}
        client.put("/api/pbx/NoWcPBX",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                     "sip_password": ""}), headers=admin)
        # Must declare an explicit feature list that excludes
        # "webclient" — empty list = permissive-all in auth.py.
        client.put("/api/accounts/NoWcAcc",
                   data=json.dumps({"token": "no-wc-tok", "pbx": "NoWcPBX",
                                     "features": ["tts", "stt"]}),
                   headers=admin)
        client.put("/api/subscribe/no-wc-sub",
                   data=json.dumps({
                       "base_url": "https://crm.example.com/crm",
                       "bearer_token": "t",
                   }), headers=acct)
        call_sid = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": "no-wc-sub"}),
                               headers=acct).get_json()["call_id"]
        try:
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": 'webclient:userA{"callback":"/cb",'
                                          '"base_url":"https://crm.example.com",'
                                          f'"call_id":"{call_sid}"}}',
                               }),
                               headers=acct)
            assert resp.status_code in (400, 403), (
                f"Account without webclient feature got "
                f"{resp.status_code} (expected 400/403): "
                f"{resp.get_data(as_text=True)}"
            )
            body = resp.get_data(as_text=True).lower()
            assert "webclient" in body or "feature" in body, (
                f"error message should mention the missing feature: "
                f"{body!r}"
            )
        finally:
            client.delete(f"/api/calls/{call_sid}", headers=acct)
