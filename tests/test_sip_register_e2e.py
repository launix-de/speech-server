"""End-to-end SIP REGISTER contract against FakeCrm.

The built-in SIP registrar must:
- reject/challenge unauthenticated device REGISTERs
- fetch HA1 from the CRM login endpoint for authenticated REGISTERs
- persist the successful registration for later INVITE routing
"""
from __future__ import annotations

import hashlib
import re

import pytest

from conftest import SUBSCRIBER_ID
from fake_crm import FakeCrm
from speech_pipeline.telephony import sip_stack, subscriber as sub_mod


def _register_message(sip_user: str, *, auth_header: str | None = None) -> dict:
    lines = [
        "REGISTER sip:srv.launix.de:5061 SIP/2.0",
        "Via: SIP/2.0/UDP 155.133.219.85:5060;branch=z9hG4bK-test",
        f"From: <sip:{sip_user}>;tag=from-1",
        f"To: <sip:{sip_user}>",
        "Call-ID: reg-test",
        "CSeq: 1 REGISTER",
        "Contact: <sip:alice@155.133.219.85:5060>",
    ]
    if auth_header:
        lines.append(f"Authorization: {auth_header}")
    lines.extend(["Content-Length: 0", "", ""])
    return sip_stack._parse_sip("\r\n".join(lines).encode("utf-8"))


@pytest.fixture(autouse=True)
def _clear_sip_state():
    sip_stack._nonces.clear()
    sip_stack._HA1_CACHE.clear()
    sip_stack._registrations.clear()
    yield
    sip_stack._nonces.clear()
    sip_stack._HA1_CACHE.clear()
    sip_stack._registrations.clear()


@pytest.fixture
def deployed_crm(client, admin, account):
    token = account["Authorization"].split(None, 1)[1]
    crm = FakeCrm(client, admin_headers=admin, account_token=token)
    crm.login_tokens["alice"] = "alice-login-token"
    crm.login_user_ids["alice"] = 42
    crm.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")

    sub = sub_mod.get(SUBSCRIBER_ID)
    assert sub is not None
    assert sub["base_url"] == crm.BASE_URL
    crm.sip_domain = sub_mod.base_url_to_sip_domain(sub["base_url"])
    assert sub["events"]["incoming"].endswith("state=incoming")
    assert sub["events"]["device_dial"].endswith("state=device-dial")
    assert sub["events"]["call_ended"].endswith("state=ended")
    return crm


class TestSipRegisterWithFakeCrm:

    def test_unauthenticated_register_is_rejected(self, deployed_crm, monkeypatch):
        sent: list[tuple[str, tuple[str, int]]] = []
        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent.append((data, addr)),
        )

        sip_user = f"alice@{deployed_crm.sip_domain}"
        msg = _register_message(sip_user)
        sip_stack._handle_inbound_register(msg, ("155.133.219.85", 5060))

        assert len(sent) == 1
        payload, addr = sent[0]
        assert addr == ("155.133.219.85", 5060)
        assert "SIP/2.0 401 Unauthorized" in payload
        assert (
            f'WWW-Authenticate: Digest realm="{deployed_crm.sip_domain}"' in payload
        )
        assert sip_stack.get_registrations(sip_user) == []
        assert deployed_crm.login_requests == []

    def test_authenticated_register_succeeds(self, deployed_crm, monkeypatch):
        sent: list[tuple[str, tuple[str, int]]] = []
        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent.append((data, addr)),
        )

        with deployed_crm.active(monkeypatch):
            sip_user = f"alice@{deployed_crm.sip_domain}"
            first = _register_message(sip_user)
            sip_stack._handle_inbound_register(first, ("155.133.219.85", 5060))

            challenge, _ = sent[-1]
            nonce_match = re.search(r'nonce="([^"]+)"', challenge)
            assert nonce_match, challenge
            nonce = nonce_match.group(1)

            realm = deployed_crm.sip_domain
            ha1 = hashlib.md5(
                f"{sip_user}:{realm}:{deployed_crm.login_tokens['alice']}".encode(
                    "utf-8"
                )
            ).hexdigest()
            digest = sip_stack._compute_digest_response_ha1(
                ha1, nonce, "REGISTER", "sip:srv.launix.de:5061"
            )
            auth = (
                "Digest "
                f'username="{sip_user}", '
                f'realm="{realm}", '
                f'nonce="{nonce}", '
                'uri="sip:srv.launix.de:5061", '
                f'response="{digest}", '
                "algorithm=MD5"
            )

            second = _register_message(sip_user, auth_header=auth)
            sip_stack._handle_inbound_register(second, ("155.133.219.85", 5060))

        payload, addr = sent[-1]
        assert addr == ("155.133.219.85", 5060)
        assert "SIP/2.0 200 OK" in payload

        regs = sip_stack.get_registrations(sip_user)
        assert len(regs) == 1
        reg = regs[0]
        assert reg["user_id"] == 42
        assert reg["subscriber_id"] == SUBSCRIBER_ID
        assert reg["base_url"] == deployed_crm.BASE_URL
        assert reg["contact_uri"] == "sip:alice@155.133.219.85:5060"
        assert reg["source_addr"] == ("155.133.219.85", 5060)

        assert len(deployed_crm.login_requests) == 1
        login = deployed_crm.login_requests[0]
        assert login["token"] == deployed_crm.account_token
        assert login["query"]["username"] == "alice"
        assert login["query"]["realm"] == deployed_crm.sip_domain
        assert login["query"]["sip_user"] == sip_user
