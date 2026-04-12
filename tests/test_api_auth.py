"""Authentication and authorization tests.

Covers: token validation, admin vs account tier, missing/malformed auth,
per-route access control for every admin-only and account-scoped endpoint.
"""
import json

from conftest import ADMIN_TOKEN, ACCOUNT_TOKEN, ACCOUNT_ID


class TestAuthTokenValidation:
    """Token extraction and rejection."""

    def test_no_auth_header_returns_401(self, client):
        assert client.get("/api/calls").status_code == 401

    def test_wrong_token_rejected(self, client):
        h = {"Authorization": "Bearer wrong", "Content-Type": "application/json"}
        assert client.put("/api/accounts/x",
                          data=json.dumps({"token": "x"}), headers=h
                          ).status_code in (401, 403)

    def test_missing_bearer_prefix_rejected(self, client):
        h = {"Authorization": ADMIN_TOKEN, "Content-Type": "application/json"}
        assert client.get("/api/accounts", headers=h).status_code in (401, 403)

    def test_empty_bearer_rejected(self, client):
        h = {"Authorization": "Bearer ", "Content-Type": "application/json"}
        assert client.get("/api/calls", headers=h).status_code in (401, 403)

    def test_bearer_with_extra_whitespace(self, client):
        h = {"Authorization": f"Bearer  {ADMIN_TOKEN}", "Content-Type": "application/json"}
        # Double space → token starts with space → not admin token
        assert client.get("/api/accounts", headers=h).status_code in (401, 403)


class TestAdminAccess:
    """Admin token grants access to admin-only routes."""

    def test_admin_can_list_accounts(self, client, admin):
        assert client.get("/api/accounts", headers=admin).status_code == 200

    def test_admin_can_list_pbx(self, client, admin):
        assert client.get("/api/pbx", headers=admin).status_code == 200

    def test_admin_can_create_account(self, client, admin):
        resp = client.put("/api/accounts/adm-test",
                          data=json.dumps({"token": "t"}),
                          headers=admin)
        assert resp.status_code == 200

    def test_admin_can_access_account_routes(self, client, admin, account):
        """Admin token also passes require_account checks."""
        resp = client.get("/api/calls", headers=admin)
        assert resp.status_code == 200


class TestAccountAccessDenied:
    """Account tokens must be rejected on admin-only routes."""

    def test_account_cannot_list_accounts(self, client, account):
        assert client.get("/api/accounts", headers=account).status_code in (401, 403)

    def test_account_cannot_get_account(self, client, account):
        assert client.get(f"/api/accounts/{ACCOUNT_ID}", headers=account).status_code == 403

    def test_account_cannot_create_account(self, client, account):
        resp = client.put("/api/accounts/evil",
                          data=json.dumps({"token": "evil-tok"}),
                          headers=account)
        assert resp.status_code == 403

    def test_account_cannot_delete_account(self, client, account):
        assert client.delete(f"/api/accounts/{ACCOUNT_ID}", headers=account).status_code == 403

    def test_account_cannot_create_pbx(self, client, account):
        resp = client.put("/api/pbx/HackPBX",
                          data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
                          headers=account)
        assert resp.status_code == 403

    def test_account_cannot_list_pbx(self, client, account):
        assert client.get("/api/pbx", headers=account).status_code == 403

    def test_account_cannot_delete_pbx(self, client, account):
        assert client.delete("/api/pbx/TestPBX", headers=account).status_code == 403


class TestNoAuthOnAccountRoutes:
    """Account-scoped routes require at least a token."""

    def test_subscribers(self, client):
        assert client.get("/api/subscribers").status_code == 401

    def test_calls(self, client):
        assert client.get("/api/calls").status_code == 401

    def test_nonce(self, client):
        assert client.post("/api/nonce",
                           data=json.dumps({"subscriber_id": "x", "user": "y"})
                           ).status_code == 401

    def test_call_detail_via_pipelines(self, client):
        assert client.get("/api/pipelines?dsl=call:call-xxx").status_code == 401

    def test_pipelines_create(self, client):
        assert client.post("/api/pipelines",
                           data=json.dumps({"dsl": "play:x"})
                           ).status_code == 401

    def test_nonces(self, client):
        assert client.get("/api/nonces").status_code == 401
