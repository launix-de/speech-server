"""Subscriber management API tests.

Covers: CRUD, DID conflicts, SIP domain mapping, ownership, idempotent
re-registration, inbound routing lookups.
"""
import json

from conftest import SUBSCRIBER_ID


class TestSubscriberCRUD:
    def test_create_subscriber(self, client, account):
        resp = client.get(f"/api/subscribers/{SUBSCRIBER_ID}", headers=account)
        assert resp.status_code == 200
        assert resp.get_json()["base_url"] == "https://example.com/crm"

    def test_list_subscribers(self, client, account):
        resp = client.get("/api/subscribers", headers=account)
        assert resp.status_code == 200
        assert len(resp.get_json()) >= 1

    def test_delete_subscriber(self, client, account):
        resp = client.delete(f"/api/subscribers/{SUBSCRIBER_ID}", headers=account)
        assert resp.status_code in (200, 204)

    def test_get_nonexistent_subscriber(self, client, account):
        resp = client.get("/api/subscribers/no-such", headers=account)
        assert resp.status_code == 404

    def test_delete_nonexistent_subscriber(self, client, account):
        resp = client.delete("/api/subscribers/no-such", headers=account)
        assert resp.status_code == 404


class TestSubscriberSIPDomain:
    def test_sip_domain_registered(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        sub = find_by_sip_domain("crm.example.com")
        assert sub is not None
        assert sub["id"] == SUBSCRIBER_ID

    def test_sip_domain_updated_on_reregister(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        client.put("/api/subscribe/sub-sip",
                   data=json.dumps({
                       "base_url": "https://old.example.com/crm",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm.old.example.com") is not None
        client.put("/api/subscribe/sub-sip",
                   data=json.dumps({
                       "base_url": "https://new.example.com/crm",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm.old.example.com") is None
        assert find_by_sip_domain("crm.new.example.com") is not None

    def test_sip_domain_deep_path(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        client.put("/api/subscribe/sub-deep",
                   data=json.dumps({
                       "base_url": "https://example.net/app/crm-next",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm-next.app.example.net") is not None

    def test_sip_domain_no_path(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        client.put("/api/subscribe/sub-nopath",
                   data=json.dumps({
                       "base_url": "https://crm.example.net",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm.example.net") is not None

    def test_base_url_lookup_is_normalized(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_base_url
        client.put("/api/subscribe/sub-baseurl",
                   data=json.dumps({
                       "base_url": "https://crm.example.com/app/",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_base_url("https://crm.example.com/app") is not None
        assert find_by_base_url("crm.example.com/app/") is not None
        assert find_by_base_url("http://crm.example.com/app") is not None

    def test_registration_target_lookup_accepts_legacy_host_path_alias(
            self, client, account):
        from speech_pipeline.telephony.subscriber import (
            find_by_registration_target, find_unique_subscriber,
        )
        client.put("/api/subscribe/sub-legacy-target",
                   data=json.dumps({
                       "base_url": "https://example.net/app/crm-next",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_registration_target("crm-next.app.example.net") is not None
        assert find_by_registration_target("example.net/app/crm-next") is not None
        assert find_unique_subscriber() is None

    def test_unique_subscriber_detects_single_live_subscriber(
            self, client, account):
        from speech_pipeline.telephony import subscriber as sub_mod
        client.delete("/api/subscribers/test-subscriber", headers=account)
        client.put("/api/subscribe/sub-only",
                   data=json.dumps({
                       "base_url": "https://only.example.com/crm",
                       "bearer_token": "t",
                   }),
                   headers=account)
        unique = sub_mod.find_unique_subscriber()
        assert unique is not None
        assert unique["id"] == "sub-only"


class TestSubscriberDIDs:
    def test_did_conflict_rejected(self, client, account):
        """Two subscribers cannot claim the same DID."""
        client.put("/api/subscribe/sub-did-a",
                   data=json.dumps({
                       "base_url": "https://a.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+4912345"],
                   }),
                   headers=account)
        resp = client.put("/api/subscribe/sub-did-b",
                          data=json.dumps({
                              "base_url": "https://b.example.com",
                              "bearer_token": "t",
                              "inbound_dids": ["+4912345"],
                          }),
                          headers=account)
        assert resp.status_code == 400

    def test_did_freed_after_delete(self, client, account):
        """Deleting subscriber frees its DIDs for reuse."""
        client.put("/api/subscribe/sub-did-free",
                   data=json.dumps({
                       "base_url": "https://free.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49999"],
                   }),
                   headers=account)
        client.delete("/api/subscribers/sub-did-free", headers=account)
        resp = client.put("/api/subscribe/sub-did-reuse",
                          data=json.dumps({
                              "base_url": "https://reuse.example.com",
                              "bearer_token": "t",
                              "inbound_dids": ["+49999"],
                          }),
                          headers=account)
        assert resp.status_code == 200

    def test_did_lookup(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_did
        client.put("/api/subscribe/sub-did-lookup",
                   data=json.dumps({
                       "base_url": "https://lookup.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49111"],
                   }),
                   headers=account)
        sub = find_by_did("+49111")
        assert sub is not None
        assert sub["id"] == "sub-did-lookup"

    def test_did_lookup_miss(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_did
        assert find_by_did("+49000000") is None

    def test_multiple_dids(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_did
        client.put("/api/subscribe/sub-multi-did",
                   data=json.dumps({
                       "base_url": "https://multi.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49222", "+49333"],
                   }),
                   headers=account)
        assert find_by_did("+49222")["id"] == "sub-multi-did"
        assert find_by_did("+49333")["id"] == "sub-multi-did"

    def test_did_update_on_reregister(self, client, account):
        """Reregistering subscriber updates DID map."""
        from speech_pipeline.telephony.subscriber import find_by_did
        client.put("/api/subscribe/sub-did-upd",
                   data=json.dumps({
                       "base_url": "https://upd.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49444"],
                   }),
                   headers=account)
        client.put("/api/subscribe/sub-did-upd",
                   data=json.dumps({
                       "base_url": "https://upd.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49555"],
                   }),
                   headers=account)
        assert find_by_did("+49444") is None
        assert find_by_did("+49555")["id"] == "sub-did-upd"


class TestSubscriberOwnership:
    def test_idempotent_reregister(self, client, account):
        resp = client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                          data=json.dumps({
                              "base_url": "https://updated.example.com/crm",
                              "bearer_token": "new-token",
                          }),
                          headers=account)
        assert resp.status_code == 200
        assert resp.get_json()["base_url"] == "https://updated.example.com/crm"

    def test_subscriber_stores_events(self, client, account):
        client.put("/api/subscribe/sub-events",
                   data=json.dumps({
                       "base_url": "https://events.example.com",
                       "bearer_token": "t",
                       "events": {"inbound": "/Telephone/Inbound"},
                   }),
                   headers=account)
        resp = client.get("/api/subscribers/sub-events", headers=account)
        assert resp.get_json()["events"]["inbound"] == "/Telephone/Inbound"


class TestSubscriberLoginUrl:
    """The CRM provisions ``login_url`` (absolute) so the speech server
    never has to construct CRM endpoint paths itself."""

    def test_login_url_round_trips_verbatim(self, client, account):
        client.put("/api/subscribe/sub-login-url",
                   data=json.dumps({
                       "base_url": "https://login.example.com/crm",
                       "bearer_token": "t",
                       "login_url": "https://login.example.com/crm/Telephone/SpeechServer/login",
                   }),
                   headers=account)
        resp = client.get("/api/subscribers/sub-login-url", headers=account)
        assert resp.get_json()["login_url"] == (
            "https://login.example.com/crm/Telephone/SpeechServer/login"
        )

    def test_login_url_can_point_to_a_different_host_than_base_url(
            self, client, account):
        """The CRM is free to expose its loginAction under a different host
        (e.g. an internal API endpoint)."""
        client.put("/api/subscribe/sub-split-host",
                   data=json.dumps({
                       "base_url": "https://crm.example.com/app",
                       "bearer_token": "t",
                       "login_url": "https://auth.example.com/sip-login",
                   }),
                   headers=account)
        resp = client.get("/api/subscribers/sub-split-host", headers=account)
        assert resp.get_json()["login_url"] == "https://auth.example.com/sip-login"

    def test_subscriber_without_login_url_stores_empty_string(
            self, client, account):
        """Provisioning a subscriber without ``login_url`` is allowed at the
        API layer; SIP REGISTER auth then hard-fails because the speech
        server has no way to validate credentials. See ``_crm_login``."""
        client.put("/api/subscribe/sub-no-login",
                   data=json.dumps({
                       "base_url": "https://nologin.example.com",
                       "bearer_token": "t",
                   }),
                   headers=account)
        resp = client.get("/api/subscribers/sub-no-login", headers=account)
        assert resp.get_json()["login_url"] == ""
