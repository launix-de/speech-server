"""Account management API tests.

Covers: CRUD, missing fields, cascade delete, features, PBX pinning.
"""
import json

from conftest import ADMIN_TOKEN


class TestAccountCRUD:
    def test_create_account(self, client, admin):
        resp = client.put("/api/accounts/acct1",
                          data=json.dumps({"token": "tok1"}),
                          headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["id"] == "acct1"

    def test_get_account(self, client, admin):
        client.put("/api/accounts/acct2",
                   data=json.dumps({"token": "tok2"}),
                   headers=admin)
        resp = client.get("/api/accounts/acct2", headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["id"] == "acct2"

    def test_list_accounts(self, client, admin):
        client.put("/api/accounts/acct-list",
                   data=json.dumps({"token": "tok-list"}),
                   headers=admin)
        resp = client.get("/api/accounts", headers=admin)
        assert resp.status_code == 200
        ids = [a["id"] for a in resp.get_json()]
        assert "acct-list" in ids

    def test_delete_account(self, client, admin):
        client.put("/api/accounts/acct-del",
                   data=json.dumps({"token": "tok-del"}),
                   headers=admin)
        resp = client.delete("/api/accounts/acct-del", headers=admin)
        assert resp.status_code in (200, 204)

    def test_update_account_overwrites(self, client, admin):
        client.put("/api/accounts/acct-up",
                   data=json.dumps({"token": "tok-old"}),
                   headers=admin)
        client.put("/api/accounts/acct-up",
                   data=json.dumps({"token": "tok-new", "pbx": "SomePBX"}),
                   headers=admin)
        resp = client.get("/api/accounts/acct-up", headers=admin)
        data = resp.get_json()
        assert data["token"] == "tok-new"
        assert data["pbx"] == "SomePBX"


class TestAccountValidation:
    def test_create_account_missing_token(self, client, admin):
        resp = client.put("/api/accounts/bad",
                          data=json.dumps({}),
                          headers=admin)
        assert resp.status_code == 400

    def test_get_nonexistent_account(self, client, admin):
        resp = client.get("/api/accounts/no-such", headers=admin)
        assert resp.status_code == 404

    def test_delete_nonexistent_account(self, client, admin):
        resp = client.delete("/api/accounts/no-such", headers=admin)
        assert resp.status_code == 404


class TestAccountCascade:
    def test_delete_account_cascades_subscribers(self, client, admin):
        """Deleting an account removes its subscribers."""
        acct_h = {"Authorization": "Bearer cascade-tok",
                  "Content-Type": "application/json"}
        client.put("/api/accounts/cascade-acct",
                   data=json.dumps({"token": "cascade-tok"}),
                   headers=admin)
        client.put("/api/subscribe/cascade-sub",
                   data=json.dumps({
                       "base_url": "https://cascade.example.com",
                       "bearer_token": "t",
                   }),
                   headers=acct_h)
        # Verify subscriber exists
        resp = client.get("/api/subscribers/cascade-sub", headers=acct_h)
        assert resp.status_code == 200
        # Delete account
        client.delete("/api/accounts/cascade-acct", headers=admin)
        from speech_pipeline.telephony.subscriber import get
        assert get("cascade-sub") is None


class TestAccountFeatures:
    def test_create_account_with_features(self, client, admin):
        resp = client.put("/api/accounts/feat-acct",
                          data=json.dumps({
                              "token": "feat-tok",
                              "features": ["tts", "webclient"],
                          }),
                          headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["features"] == ["tts", "webclient"]

    def test_create_account_with_max_concurrent(self, client, admin):
        resp = client.put("/api/accounts/conc-acct",
                          data=json.dumps({
                              "token": "conc-tok",
                              "max_concurrent_calls": 10,
                          }),
                          headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["max_concurrent_calls"] == 10

    def test_account_no_features_means_all(self, client, admin):
        """Empty features list = all features allowed."""
        from speech_pipeline.telephony.auth import check_feature
        client.put("/api/accounts/open-acct",
                   data=json.dumps({"token": "open-tok"}),
                   headers=admin)
        assert check_feature("open-acct", "tts") is True
        assert check_feature("open-acct", "stt") is True

    def test_account_restricted_features(self, client, admin):
        from speech_pipeline.telephony.auth import check_feature
        client.put("/api/accounts/restricted",
                   data=json.dumps({"token": "restr-tok", "features": ["tts"]}),
                   headers=admin)
        assert check_feature("restricted", "tts") is True
        assert check_feature("restricted", "stt") is False


class TestAccountPBXAccess:
    def test_pbx_pin_allows_pinned_pbx(self, client, admin):
        from speech_pipeline.telephony.auth import check_pbx_access
        client.put("/api/accounts/pin-acct",
                   data=json.dumps({"token": "pin-tok", "pbx": "MyPBX"}),
                   headers=admin)
        assert check_pbx_access("pin-acct", "MyPBX") is True

    def test_pbx_pin_rejects_other_pbx(self, client, admin):
        from speech_pipeline.telephony.auth import check_pbx_access
        client.put("/api/accounts/pin-acct2",
                   data=json.dumps({"token": "pin-tok2", "pbx": "MyPBX"}),
                   headers=admin)
        assert check_pbx_access("pin-acct2", "OtherPBX") is False

    def test_no_pbx_pin_allows_any(self, client, admin):
        from speech_pipeline.telephony.auth import check_pbx_access
        client.put("/api/accounts/free-acct",
                   data=json.dumps({"token": "free-tok"}),
                   headers=admin)
        assert check_pbx_access("free-acct", "AnyPBX") is True
