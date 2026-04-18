"""End-to-end SIP REGISTER contract against FakeCrm.

The built-in SIP registrar must:
- reject/challenge unauthenticated device REGISTERs
- fetch HA1 from the CRM login endpoint for authenticated REGISTERs
- persist the successful registration for later INVITE routing
"""
from __future__ import annotations

import hashlib
import re
import threading

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


def _invite_message(from_sip_user: str, target_number: str) -> dict:
    body = "\r\n".join([
        "v=0",
        "o=user 1 1 IN IP4 10.0.0.99",
        "s=-",
        "c=IN IP4 10.0.0.99",
        "t=0 0",
        "m=audio 4000 RTP/AVP 0 101",
        "a=rtpmap:0 PCMU/8000",
        "a=rtpmap:101 telephone-event/8000",
        "a=fmtp:101 0-15",
        "",
    ])
    lines = [
        f"INVITE sip:{target_number}@srv.launix.de SIP/2.0",
        "Via: SIP/2.0/UDP 10.0.0.99:5070;branch=z9hG4bK-invite",
        "Max-Forwards: 70",
        f"From: <sip:{from_sip_user}>;tag=from-2",
        f"To: <sip:{target_number}@srv.launix.de>",
        "Call-ID: invite-test",
        "CSeq: 1 INVITE",
        "Contact: <sip:alice@10.0.0.99:5070>",
        "Content-Type: application/sdp",
        f"Content-Length: {len(body.encode('utf-8'))}",
        "",
        body,
    ]
    return sip_stack._parse_sip("\r\n".join(lines).encode("utf-8"))


@pytest.fixture(autouse=True)
def _clear_sip_state():
    sip_stack._nonces.clear()
    sip_stack._HA1_CACHE.clear()
    sip_stack._registrations.clear()
    sip_stack._trunk_dialogs.clear()
    yield
    sip_stack._nonces.clear()
    sip_stack._HA1_CACHE.clear()
    sip_stack._registrations.clear()
    sip_stack._trunk_dialogs.clear()


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

    def test_registered_client_invite_uses_full_aor_and_fires_device_dial(
            self, deployed_crm, monkeypatch):
        import speech_pipeline.RTPSession as rtp_mod
        from speech_pipeline.telephony import dispatcher, sip_listener

        sent: list[tuple[str, tuple[str, int]]] = []
        events: list[tuple[dict, str, dict]] = []

        class _DummyRtpSession:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        class _DummyCallSession:
            def __init__(self, _rtp):
                self.hungup = threading.Event()

        monkeypatch.setattr(sip_stack, "_send",
                            lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_find_free_port", lambda: 16000)
        monkeypatch.setattr(rtp_mod, "RTPSession", _DummyRtpSession)
        monkeypatch.setattr(rtp_mod, "RTPCallSession", _DummyCallSession)
        monkeypatch.setattr(dispatcher, "fire_subscriber_event",
                            lambda sub, key, payload: events.append(
                                (sub, key, payload)
                            ) or [])
        monkeypatch.setattr(sip_listener, "_wait_for_bridge",
                            lambda *args, **kwargs: None)

        sip_user = f"alice@{deployed_crm.sip_domain}"
        with deployed_crm.active(monkeypatch):
            first = _register_message(sip_user)
            sip_stack._handle_inbound_register(first, ("10.0.0.5", 5060))

            challenge, _ = sent[-1]
            nonce = re.search(r'nonce="([^"]+)"', challenge).group(1)
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
            sip_stack._handle_inbound_register(second, ("10.0.0.5", 5060))

            invite = _invite_message(sip_user, "+4930123456")
            sip_stack._handle_inbound_invite(invite, ("10.0.0.99", 5070))

        assert events, (
            "registered-client INVITE did not fire device_dial; "
            "the server likely failed to resolve the source registration"
        )
        sub, key, payload = events[0]
        assert sub["id"] == SUBSCRIBER_ID
        assert key == "device_dial"
        assert payload["number"] == "+4930123456"
        assert payload["sip_user"] == sip_user
        assert any("SIP/2.0 183 Session Progress" in data for data, _ in sent), (
            "registered-client INVITE was not accepted with 183 Session Progress"
        )
        assert not any("SIP/2.0 404 Not Found" in data for data, _ in sent), (
            "server misclassified a registered client INVITE as an unregistered target"
        )

        assert len(deployed_crm.login_requests) == 1
        login = deployed_crm.login_requests[0]
        assert login["token"] == deployed_crm.account_token
        assert login["query"]["username"] == "alice"
        assert login["query"]["realm"] == deployed_crm.sip_domain
        assert login["query"]["sip_user"] == sip_user
