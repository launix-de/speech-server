"""Nonce management API tests.

Covers: CRUD, missing fields, nonexistent subscriber, TTL expiry,
nonce format, list filtering.
"""
import json
import time

from conftest import SUBSCRIBER_ID, create_call


class TestNonceCRUD:
    def test_create_nonce(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "user": "testuser",
                           }),
                           headers=account)
        assert resp.status_code in (200, 201)
        data = resp.get_json()
        assert "nonce" in data
        assert data["nonce"].startswith("n-")

    def test_list_nonces(self, client, account):
        client.post("/api/nonce",
                    data=json.dumps({
                        "subscriber_id": SUBSCRIBER_ID,
                        "user": "testuser",
                    }),
                    headers=account)
        resp = client.get("/api/nonces", headers=account)
        assert resp.status_code == 200
        assert len(resp.get_json()) >= 1

    def test_revoke_nonce(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "user": "u",
                           }),
                           headers=account)
        nonce = resp.get_json()["nonce"]
        resp = client.delete(f"/api/nonce/{nonce}", headers=account)
        assert resp.status_code in (200, 204)


class TestNonceValidation:
    def test_missing_user(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                           headers=account)
        assert resp.status_code == 400

    def test_missing_subscriber_id(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({"user": "u"}),
                           headers=account)
        assert resp.status_code == 400

    def test_nonexistent_subscriber(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": "ghost",
                               "user": "u",
                           }),
                           headers=account)
        assert resp.status_code == 404

    def test_revoke_nonexistent_nonce(self, client, account):
        resp = client.delete("/api/nonce/n-does-not-exist", headers=account)
        assert resp.status_code == 404


class TestNonceTTL:
    def test_nonce_expired(self, client, account):
        """Nonce with TTL=0 should be expired immediately."""
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "user": "u",
                               "ttl": 0,
                           }),
                           headers=account)
        assert resp.status_code == 201
        nonce = resp.get_json()["nonce"]
        time.sleep(0.01)
        from speech_pipeline.telephony.auth import validate_nonce
        assert validate_nonce(nonce) is None

    def test_nonce_valid_within_ttl(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "user": "u",
                               "ttl": 3600,
                           }),
                           headers=account)
        nonce = resp.get_json()["nonce"]
        from speech_pipeline.telephony.auth import validate_nonce
        entry = validate_nonce(f"test-account:{nonce}")
        assert entry is not None
        assert entry["user"] == "u"


class TestNonceMetadata:
    def test_nonce_stores_subscriber_id(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "user": "u",
                           }),
                           headers=account)
        data = resp.get_json()
        assert data["subscriber_id"] == SUBSCRIBER_ID

    def test_nonce_stores_user(self, client, account):
        resp = client.post("/api/nonce",
                           data=json.dumps({
                               "subscriber_id": SUBSCRIBER_ID,
                               "user": "agent42",
                           }),
                           headers=account)
        assert resp.get_json()["user"] == "agent42"

    def test_multiple_nonces_listed(self, client, account):
        for i in range(3):
            client.post("/api/nonce",
                        data=json.dumps({
                            "subscriber_id": SUBSCRIBER_ID,
                            "user": f"user{i}",
                        }),
                        headers=account)
        resp = client.get("/api/nonces", headers=account)
        assert len(resp.get_json()) >= 3
