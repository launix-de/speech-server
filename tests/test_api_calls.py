"""Call lifecycle API tests.

Covers: CRUD, participants, PBX pinning, direction/metadata, nonexistent
subscriber, double-delete.
"""
import json

from conftest import SUBSCRIBER_ID, create_call


class TestCallCRUD:
    def test_create_call(self, client, account):
        call_id = create_call(client, account)
        assert call_id.startswith("call-")

    def test_list_calls(self, client, account):
        create_call(client, account)
        resp = client.get("/api/calls", headers=account)
        assert resp.status_code == 200
        assert len(resp.get_json()) >= 1

    def test_get_call(self, client, account):
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.status_code == 200
        assert resp.get_json()["call_id"] == call_id

    def test_delete_call(self, client, account):
        call_id = create_call(client, account)
        resp = client.delete(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 204
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.status_code == 404

    def test_delete_nonexistent_call(self, client, account):
        resp = client.delete("/api/calls/no-such-call", headers=account)
        assert resp.status_code == 404

    def test_get_nonexistent_call(self, client, account):
        resp = client.get("/api/pipelines?dsl=call:no-such-call", headers=account)
        assert resp.status_code == 404


class TestCallParticipants:
    def test_list_participants_empty(self, client, account):
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.status_code == 200
        assert resp.get_json()["participants"] == []

    def test_participants_nonexistent_call(self, client, account):
        resp = client.get("/api/pipelines?dsl=call:no-such", headers=account)
        assert resp.status_code == 404


class TestCallValidation:
    def test_create_call_missing_subscriber(self, client, account):
        resp = client.post("/api/calls",
                           data=json.dumps({"subscriber_id": "nonexistent"}),
                           headers=account)
        assert resp.status_code == 404

    def test_create_call_pbx_pin_conflict(self, client, admin, account):
        """Account pinned to PBX A cannot request PBX B."""
        resp = client.post("/api/calls",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "pbx": "OtherPBX",
                           }),
                           headers=account)
        assert resp.status_code == 403


class TestCallMetadata:
    def test_call_direction_stored(self, client, account):
        resp = client.post("/api/calls",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "direction": "outbound",
                               "caller": "+491111",
                               "callee": "+492222",
                           }),
                           headers=account)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["direction"] == "outbound"
        assert data["caller"] == "+491111"
        assert data["callee"] == "+492222"

    def test_call_default_direction_inbound(self, client, account):
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.get_json()["direction"] == "inbound"

    def test_call_status_active(self, client, account):
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.get_json()["status"] == "active"

    def test_call_subscriber_stored(self, client, account):
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert resp.get_json()["subscriber_id"] == SUBSCRIBER_ID

    def test_call_has_created_at(self, client, account):
        call_id = create_call(client, account)
        resp = client.get(f"/api/pipelines?dsl=call:{call_id}", headers=account)
        assert "created_at" in resp.get_json()
        assert isinstance(resp.get_json()["created_at"], float)


class TestCallMultiple:
    def test_multiple_calls_listed(self, client, account):
        ids = [create_call(client, account) for _ in range(3)]
        resp = client.get("/api/calls", headers=account)
        listed_ids = [c["call_id"] for c in resp.get_json()]
        for cid in ids:
            assert cid in listed_ids

    def test_delete_one_does_not_affect_others(self, client, account):
        id1 = create_call(client, account)
        id2 = create_call(client, account)
        client.delete(f"/api/calls/{id1}", headers=account)
        resp = client.get(f"/api/pipelines?dsl=call:{id2}", headers=account)
        assert resp.status_code == 200
