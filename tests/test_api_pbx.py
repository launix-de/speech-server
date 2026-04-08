"""PBX management API tests.

Covers: CRUD, idempotent update, password filtering, nonexistent delete.
"""
import json


class TestPBXCRUD:
    def test_create_pbx(self, client, admin):
        resp = client.put("/api/pbx/MyPBX",
                          data=json.dumps({"sip_proxy": "sip.example.com",
                                           "sip_user": "user", "sip_password": "pass"}),
                          headers=admin)
        assert resp.status_code == 200

    def test_list_pbx(self, client, admin):
        client.put("/api/pbx/PBX1",
                   data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
                   headers=admin)
        resp = client.get("/api/pbx", headers=admin)
        assert resp.status_code == 200
        assert len(resp.get_json()) >= 1

    def test_delete_pbx(self, client, admin):
        client.put("/api/pbx/PBX-del",
                   data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
                   headers=admin)
        resp = client.delete("/api/pbx/PBX-del", headers=admin)
        assert resp.status_code in (200, 204)

    def test_delete_nonexistent_pbx(self, client, admin):
        resp = client.delete("/api/pbx/no-such", headers=admin)
        assert resp.status_code == 404


class TestPBXEdgeCases:
    def test_pbx_update_idempotent(self, client, admin):
        for _ in range(2):
            resp = client.put("/api/pbx/IdempPBX",
                              data=json.dumps({"sip_proxy": "sip.example.com",
                                               "sip_user": "u", "sip_password": "p"}),
                              headers=admin)
            assert resp.status_code == 200

    def test_pbx_list_hides_passwords(self, client, admin):
        client.put("/api/pbx/SecretPBX",
                   data=json.dumps({"sip_proxy": "sip.example.com",
                                    "sip_user": "u", "sip_password": "topsecret",
                                    "ari_password": "ariscrt"}),
                   headers=admin)
        resp = client.get("/api/pbx", headers=admin)
        for entry in resp.get_json():
            if entry["id"] == "SecretPBX":
                assert "sip_password" not in entry
                assert "ari_password" not in entry
                break
        else:
            raise AssertionError("SecretPBX not found in list")

    def test_pbx_stores_all_fields(self, client, admin):
        resp = client.put("/api/pbx/FullPBX",
                          data=json.dumps({
                              "sip_proxy": "sip.test.com",
                              "sip_host": "10.0.0.1",
                              "sip_port": 5061,
                              "sip_user": "myuser",
                              "sip_password": "mypass",
                              "ari_url": "https://ari.test.com",
                              "ari_user": "ariuser",
                              "ari_password": "aripass",
                          }),
                          headers=admin)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["sip_proxy"] == "sip.test.com"
        assert data["sip_host"] == "10.0.0.1"
        assert data["sip_port"] == 5061
