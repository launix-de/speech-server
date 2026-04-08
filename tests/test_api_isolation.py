"""Cross-account isolation tests.

Account A must never see or modify account B's resources.
Admin can access everything.
"""
import json

from conftest import (
    SUBSCRIBER_ID, SUBSCRIBER2_ID, ACCOUNT2_ID,
    create_call,
)


class TestCrossAccountSubscribers:
    def test_cannot_get_other_subscriber(self, client, account, account2):
        resp = client.get(f"/api/subscribers/{SUBSCRIBER2_ID}", headers=account)
        assert resp.status_code == 403

    def test_cannot_delete_other_subscriber(self, client, account, account2):
        resp = client.delete(f"/api/subscribers/{SUBSCRIBER2_ID}", headers=account)
        assert resp.status_code == 403

    def test_cannot_list_other_subscribers(self, client, account, account2):
        resp = client.get("/api/subscribers", headers=account)
        sub_ids = [s["id"] for s in resp.get_json()]
        assert SUBSCRIBER2_ID not in sub_ids

    def test_cannot_hijack_subscriber(self, client, account, account2):
        """Account A cannot re-register account B's subscriber."""
        resp = client.put(f"/api/subscribe/{SUBSCRIBER2_ID}",
                          data=json.dumps({
                              "base_url": "https://evil.example.com",
                              "bearer_token": "stolen",
                          }),
                          headers=account)
        assert resp.status_code == 403


class TestCrossAccountCalls:
    def test_cannot_create_call_for_other_subscriber(self, client, account, account2):
        resp = client.post("/api/calls",
                           data=json.dumps({"subscriber_id": SUBSCRIBER2_ID}),
                           headers=account)
        assert resp.status_code == 403

    def test_cannot_get_other_call(self, client, account, account2):
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.get(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 403

    def test_cannot_delete_other_call(self, client, account, account2):
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.delete(f"/api/calls/{call_id}", headers=account)
        assert resp.status_code == 403

    def test_cannot_list_other_calls(self, client, account, account2):
        create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.get("/api/calls", headers=account)
        for c in resp.get_json():
            assert c["account_id"] != ACCOUNT2_ID

    def test_cannot_get_other_call_participants(self, client, account, account2):
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.get(f"/api/calls/{call_id}/participants", headers=account)
        assert resp.status_code == 403


class TestCrossAccountPipes:
    def test_cannot_post_pipes_on_other_call(self, client, account, account2):
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": []}),
                           headers=account)
        assert resp.status_code == 403

    def test_cannot_post_commands_on_other_call(self, client, account, account2):
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.post(f"/api/calls/{call_id}/commands",
                           data=json.dumps({"commands": []}),
                           headers=account)
        assert resp.status_code == 403

    def test_cannot_delete_other_call_stage(self, client, account, account2):
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.delete(f"/api/calls/{call_id}/stages/play:x", headers=account)
        assert resp.status_code == 403


class TestCrossAccountNonces:
    def test_cannot_create_nonce_for_other_subscriber(self, client, account, account2):
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER2_ID,
                               "user": "evil",
                           }),
                           headers=account)
        assert resp.status_code == 403

    def test_nonces_filtered_per_account(self, client, account, account2):
        """Each account only sees its own nonces."""
        client.post("/api/nonce",
                    data=json.dumps({
                        "subscriber_id": SUBSCRIBER_ID,
                        "user": "u1",
                    }),
                    headers=account)
        client.post("/api/nonce",
                    data=json.dumps({
                        "subscriber_id": SUBSCRIBER2_ID,
                        "user": "u2",
                    }),
                    headers=account2)
        resp1 = client.get("/api/nonces", headers=account)
        resp2 = client.get("/api/nonces", headers=account2)
        users1 = [n["user"] for n in resp1.get_json()]
        users2 = [n["user"] for n in resp2.get_json()]
        assert "u1" in users1
        assert "u2" not in users1
        assert "u2" in users2
        assert "u1" not in users2


class TestAdminBypassesIsolation:
    def test_admin_can_access_any_call(self, client, admin, account):
        call_id = create_call(client, account)
        resp = client.get(f"/api/calls/{call_id}", headers=admin)
        assert resp.status_code == 200

    def test_admin_can_list_all_calls(self, client, admin, account, account2):
        create_call(client, account)
        create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.get("/api/calls", headers=admin)
        assert len(resp.get_json()) >= 2

    def test_admin_can_list_all_subscribers(self, client, admin, account, account2):
        resp = client.get("/api/subscribers", headers=admin)
        sub_ids = [s["id"] for s in resp.get_json()]
        assert SUBSCRIBER_ID in sub_ids
        assert SUBSCRIBER2_ID in sub_ids

    def test_admin_can_delete_any_call(self, client, admin, account):
        call_id = create_call(client, account)
        resp = client.delete(f"/api/calls/{call_id}", headers=admin)
        assert resp.status_code == 204

    def test_admin_can_list_all_nonces(self, client, admin, account, account2):
        client.post("/api/nonce",
                    data=json.dumps({"subscriber_id": SUBSCRIBER_ID, "user": "a"}),
                    headers=account)
        client.post("/api/nonce",
                    data=json.dumps({"subscriber_id": SUBSCRIBER2_ID, "user": "b"}),
                    headers=account2)
        resp = client.get("/api/nonces", headers=admin)
        users = [n["user"] for n in resp.get_json()]
        assert "a" in users
        assert "b" in users
