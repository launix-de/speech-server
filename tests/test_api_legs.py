"""Legs API tests.

Covers: list, get, delete, answer, bridge, originate — including
validation, nonexistent legs, and access control.

Note: Most leg operations require a real SIP stack. We test the API
contract (validation, 404s, auth) not the SIP signaling.
"""
import json

from conftest import SUBSCRIBER_ID, SUBSCRIBER2_ID, create_call


def _register_leg(subscriber_id=SUBSCRIBER_ID, pbx_id="TestPBX"):
    """Register a fake leg directly in the leg registry (no SIP needed)."""
    from speech_pipeline.telephony.leg import Leg, _legs
    import secrets
    leg_id = "leg-" + secrets.token_urlsafe(12)
    leg = Leg(leg_id, "inbound", "+4917012345", pbx_id, subscriber_id)
    _legs[leg_id] = leg
    return leg


class TestLegsListGet:
    def test_list_legs_empty(self, client, account):
        resp = client.get("/api/legs", headers=account)
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_list_legs_with_leg(self, client, account):
        leg = _register_leg()
        resp = client.get("/api/legs", headers=account)
        assert resp.status_code == 200
        ids = [l["leg_id"] for l in resp.get_json()]
        assert leg.leg_id in ids

    def test_get_leg(self, client, account):
        leg = _register_leg()
        resp = client.get(f"/api/legs/{leg.leg_id}", headers=account)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["leg_id"] == leg.leg_id
        assert data["direction"] == "inbound"
        assert data["number"] == "+4917012345"

    def test_get_nonexistent_leg(self, client, account):
        resp = client.get("/api/legs/leg-nosuch", headers=account)
        assert resp.status_code == 404


class TestLegsDelete:
    def test_delete_leg(self, client, account):
        leg = _register_leg()
        resp = client.delete(f"/api/legs/{leg.leg_id}", headers=account)
        assert resp.status_code in (200, 204)
        # Verify gone
        resp = client.get(f"/api/legs/{leg.leg_id}", headers=account)
        assert resp.status_code == 404

    def test_delete_nonexistent_leg(self, client, account):
        resp = client.delete("/api/legs/leg-nosuch", headers=account)
        assert resp.status_code == 404


class TestLegsAnswer:
    def test_answer_nonexistent_leg(self, client, account):
        resp = client.post("/api/legs/leg-nosuch/answer", headers=account)
        assert resp.status_code == 404

    def test_answer_leg_no_sip(self, client, account):
        """Leg without voip_call or sip_call → cannot answer."""
        leg = _register_leg()
        resp = client.post(f"/api/legs/{leg.leg_id}/answer", headers=account)
        assert resp.status_code == 400


class TestLegsBridge:
    def test_bridge_nonexistent_leg(self, client, account):
        resp = client.post("/api/legs/leg-nosuch/bridge",
                           data=json.dumps({}),
                           headers=account)
        assert resp.status_code == 404

    def test_bridge_nonexistent_call(self, client, account):
        leg = _register_leg()
        resp = client.post(f"/api/legs/{leg.leg_id}/bridge",
                           data=json.dumps({"call_id": "call-nosuch"}),
                           headers=account)
        assert resp.status_code == 404


class TestLegsOriginate:
    def test_originate_nonexistent_call(self, client, account):
        resp = client.post("/api/legs/originate",
                           data=json.dumps({
                               "call_id": "call-nosuch",
                               "to": "+4917099999",
                           }),
                           headers=account)
        assert resp.status_code == 404

    def test_originate_missing_to(self, client, account):
        call_id = create_call(client, account)
        resp = client.post("/api/legs/originate",
                           data=json.dumps({"call_id": call_id}),
                           headers=account)
        assert resp.status_code == 400

    def test_originate_cross_account_rejected(self, client, account, account2):
        """Account A cannot originate into account B's call."""
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.post("/api/legs/originate",
                           data=json.dumps({
                               "call_id": call_id,
                               "to": "+4917099999",
                           }),
                           headers=account)
        assert resp.status_code == 403


class TestLegMetadata:
    def test_leg_to_dict(self, client, account):
        leg = _register_leg()
        resp = client.get(f"/api/legs/{leg.leg_id}", headers=account)
        data = resp.get_json()
        assert "leg_id" in data
        assert "direction" in data
        assert "number" in data
        assert "pbx_id" in data
        assert "subscriber_id" in data
        assert "status" in data
        assert "created_at" in data
        assert data["status"] == "ringing"
