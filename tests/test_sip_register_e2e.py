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

CRM_BASE_URL_CASES = [
    "https://example.test",
    "https://example.test/crm",
    "https://example.test/fop/support",
    "https://north.example.net/tenant-a",
    "https://voice.example.org/division/alpha",
    "https://customer-01.example.com",
    "https://pbx.internal.example.com/site-7",
    "https://europe.example.co.uk/crm/main",
    "https://ops.example.io/team/red",
    "https://service.example.dev/path/with/depth",
]


def _register_message(
    sip_user: str,
    *,
    auth_header: str | None = None,
    auth_header_name: str = "Authorization",
) -> dict:
    lines = [
        "REGISTER sip:sip-proxy.example.test:5061 SIP/2.0",
        "Via: SIP/2.0/UDP 203.0.113.85:5060;branch=z9hG4bK-test",
        f"From: <sip:{sip_user}>;tag=from-1",
        f"To: <sip:{sip_user}>",
        "Call-ID: reg-test",
        "CSeq: 1 REGISTER",
        "Contact: <sip:alice@203.0.113.85:5060>",
    ]
    if auth_header:
        lines.append(f"{auth_header_name}: {auth_header}")
    lines.extend(["Content-Length: 0", "", ""])
    msg = sip_stack._parse_sip("\r\n".join(lines).encode("utf-8"))
    msg["_source_addr"] = ("203.0.113.85", 5060)
    return msg


def _avm_register_message(
    auth_username: str,
    *,
    auth_header: str | None = None,
    auth_header_name: str = "Proxy-Authorization",
    cseq: int = 586,
    expires: int = 1800,
) -> dict:
    lines = [
        "REGISTER sip:sip-proxy.example.test SIP/2.0",
        "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-avm",
        "Route: <sip:sip-proxy.example.test:5061;lr>",
        "From: <sip:1@sip-proxy.example.test>;tag=avm-tag",
        "To: <sip:1@sip-proxy.example.test>",
        "Call-ID: avm-reg-test",
        f"CSeq: {cseq} REGISTER",
        "Contact: <sip:1@203.0.113.85;uniq=DEVICE1>",
        f"Expires: {expires}",
        "User-Agent: AVM FRITZ!Box 7430 test",
    ]
    if auth_header:
        lines.append(f"{auth_header_name}: {auth_header}")
    lines.extend(["Content-Length: 0", "", ""])
    msg = sip_stack._parse_sip("\r\n".join(lines).encode("utf-8"))
    msg["_source_addr"] = ("203.0.113.85", 5060)
    return msg


def _challenge_params(payload: str) -> dict:
    match = re.search(r"^WWW-Authenticate: (Digest .+)$", payload, re.MULTILINE)
    assert match, payload
    return sip_stack._parse_www_authenticate(match.group(1))


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
        f"INVITE sip:{target_number}@sip-proxy.example.test SIP/2.0",
        "Via: SIP/2.0/UDP 10.0.0.99:5070;branch=z9hG4bK-invite",
        "Max-Forwards: 70",
        f"From: <sip:{from_sip_user}>;tag=from-2",
        f"To: <sip:{target_number}@sip-proxy.example.test>",
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
    sip_stack._register_challenges.clear()
    sip_stack._registrations.clear()
    sip_stack._trunk_dialogs.clear()
    yield
    sip_stack._nonces.clear()
    sip_stack._HA1_CACHE.clear()
    sip_stack._register_challenges.clear()
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

    @pytest.mark.parametrize("base_url", CRM_BASE_URL_CASES)
    def test_register_paths_work_for_many_crm_urls(
            self, client, admin, account, monkeypatch, base_url):
        token = account["Authorization"].split(None, 1)[1]
        crm = FakeCrm(client, admin_headers=admin, account_token=token)
        crm.BASE_URL = base_url
        crm.login_tokens["alice"] = "alice-login-token"
        crm.login_user_ids["alice"] = 42
        crm.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")

        sub = sub_mod.get(SUBSCRIBER_ID)
        assert sub is not None
        sip_domain = sub_mod.base_url_to_sip_domain(base_url)
        assert sub["base_url"] == base_url
        assert sip_domain

        sent: list[tuple[str, tuple[str, int]]] = []
        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent.append((data, addr)),
        )

        with crm.active(monkeypatch):
            sip_user = f"alice@{sip_domain}"
            first = _register_message(sip_user)
            sip_stack._handle_inbound_register(first, ("203.0.113.85", 5060))

            challenge = sent[-1][0]
            params = _challenge_params(challenge)
            assert params["realm"] == sip_domain
            assert "qop" not in params
            assert "opaque" not in params

            auth = sip_stack._build_authorization(
                "Authorization",
                sip_user,
                crm.login_tokens["alice"],
                params,
                "REGISTER",
                "sip:sip-proxy.example.test:5061",
            )
            second = _register_message(
                sip_user,
                auth_header=auth,
                auth_header_name="Authorization",
            )
            sip_stack._handle_inbound_register(second, ("203.0.113.85", 5060))

        assert "SIP/2.0 200 OK" in sent[-1][0]
        regs = sip_stack.get_registrations(f"alice@{sip_domain}")
        assert len(regs) == 1
        assert regs[0]["base_url"] == base_url

    def test_resolve_direct_old_style_base_url_syntax_via_normalized_key(
            self, deployed_crm):
        direct_uri = f"sip:alice@{deployed_crm.BASE_URL.removeprefix('https://')}"
        username, realm, subscriber = sip_stack._resolve_sip_identity(
            direct_uri
        )
        assert username == "alice"
        assert realm == deployed_crm.BASE_URL.removeprefix("https://")
        assert subscriber is not None
        assert subscriber["id"] == SUBSCRIBER_ID

    def test_normalize_old_proxy_style_sip_user_to_canonical_aor(
            self, deployed_crm):
        base = deployed_crm.BASE_URL.removeprefix("https://")
        old_encoded = f"alice%40{base}@sip-proxy.example.test"
        old_mangled = f"alice:{base.replace('/', '~')}@sip-proxy.example.test"
        canonical = f"alice@{deployed_crm.sip_domain}"

        assert sip_stack._split_sip_user(old_encoded) == (
            "alice", deployed_crm.BASE_URL
        )
        assert sip_stack._split_sip_user(old_mangled) == (
            "alice", deployed_crm.BASE_URL
        )
        assert sip_stack._normalize_sip_user(old_encoded) == canonical
        assert sip_stack._normalize_sip_user(old_mangled) == canonical

    def test_unauthenticated_register_is_rejected(self, deployed_crm, monkeypatch):
        sent: list[tuple[str, tuple[str, int]]] = []
        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent.append((data, addr)),
        )

        sip_user = f"alice@{deployed_crm.sip_domain}"
        msg = _register_message(sip_user)
        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert len(sent) == 1
        payload, addr = sent[0]
        assert addr == ("203.0.113.85", 5060)
        assert "SIP/2.0 401 Unauthorized" in payload
        params = _challenge_params(payload)
        assert params["realm"] == deployed_crm.sip_domain
        assert "qop" not in params
        assert "opaque" not in params
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
            sip_stack._handle_inbound_register(first, ("203.0.113.85", 5060))

            challenge, _ = sent[-1]
            params = _challenge_params(challenge)
            auth = sip_stack._build_authorization(
                "Authorization",
                sip_user,
                deployed_crm.login_tokens["alice"],
                params,
                "REGISTER",
                "sip:sip-proxy.example.test:5061",
            )

            second = _register_message(
                sip_user,
                auth_header=auth,
                auth_header_name="Authorization",
            )
            sip_stack._handle_inbound_register(second, ("203.0.113.85", 5060))

        payload, addr = sent[-1]
        assert addr == ("203.0.113.85", 5060)
        assert "SIP/2.0 200 OK" in payload

        regs = sip_stack.get_registrations(sip_user)
        assert len(regs) == 1
        reg = regs[0]
        assert reg["user_id"] == 42
        assert reg["subscriber_id"] == SUBSCRIBER_ID
        assert reg["base_url"] == deployed_crm.BASE_URL
        assert reg["contact_uri"] == "sip:alice@203.0.113.85:5060"
        assert reg["source_addr"] == ("203.0.113.85", 5060)

    def test_avm_style_register_recovers_from_stale_nonce_and_can_be_called(
            self, deployed_crm, monkeypatch):
        sent: list[tuple[str, tuple[str, int]]] = []
        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent.append((data, addr)),
        )
        monkeypatch.setattr(sip_stack, "_find_free_rtp_port", lambda: 24000)
        monkeypatch.setattr(sip_stack, "_gen_call_id", lambda: "call-avm")
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-avm")
        monkeypatch.setattr(sip_stack, "_gen_branch", lambda: "z9hG4bK-avm-out")

        sip_user = f"alice@{deployed_crm.sip_domain}"
        with deployed_crm.active(monkeypatch):
            first = _avm_register_message(auth_username=sip_user)
            sip_stack._handle_inbound_register(first, ("203.0.113.85", 5060))

            first_challenge = sent[-1][0]
            first_params = _challenge_params(first_challenge)
            first_nonce = first_params["nonce"]

            stale_auth = (
                "Digest "
                f'username="{sip_user}", '
                f'realm="{first_params["realm"]}", '
                'nonce="stale-nonce", '
                'uri="sip:sip-proxy.example.test", '
                'response="deadbeef", '
                "algorithm=MD5"
            )
            stale = _avm_register_message(
                auth_username=sip_user,
                auth_header=stale_auth,
            )
            sip_stack._handle_inbound_register(stale, ("203.0.113.85", 5060))

            stale_reply = sent[-1][0]
            assert "SIP/2.0 401 Unauthorized" in stale_reply
            retry_params = _challenge_params(stale_reply)
            retry_nonce = retry_params["nonce"]
            assert retry_nonce not in {"stale-nonce", first_nonce}

            retry_auth = sip_stack._build_authorization(
                "Authorization",
                sip_user,
                deployed_crm.login_tokens["alice"],
                retry_params,
                "REGISTER",
                "sip:sip-proxy.example.test",
            )
            retry = _avm_register_message(
                auth_username=sip_user,
                auth_header=retry_auth,
            )
            sip_stack._handle_inbound_register(retry, ("203.0.113.85", 5060))

        assert "SIP/2.0 200 OK" in sent[-1][0]
        regs = sip_stack.get_registrations(sip_user)
        assert len(regs) == 1
        reg = regs[0]
        assert reg["contact_uri"] == "sip:1@203.0.113.85;uniq=DEVICE1"
        assert reg["source_addr"] == ("203.0.113.85", 5060)

        before_invite = len(sent)
        sip_stack.call_registered_user(sip_user)
        invite_payload, invite_addr = sent[-1]
        assert len(sent) == before_invite + 1
        assert invite_addr == ("203.0.113.85", 5060)
        assert "INVITE sip:1@203.0.113.85;uniq=DEVICE1 SIP/2.0" in invite_payload
        assert "Call-ID: call-avm" in invite_payload
        assert "Contact: <sip:" in invite_payload

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
            realm = re.search(r'realm="([^"]+)"', challenge).group(1)
            ha1 = hashlib.md5(
                f"{sip_user}:{realm}:{deployed_crm.login_tokens['alice']}".encode(
                    "utf-8"
                )
            ).hexdigest()
            digest = sip_stack._compute_digest_response_ha1(
                ha1, nonce, "REGISTER", "sip:sip-proxy.example.test:5061"
            )
            auth = (
                "Digest "
                f'username="{sip_user}", '
                f'realm="{realm}", '
                f'nonce="{nonce}", '
                'uri="sip:sip-proxy.example.test:5061", '
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

    @pytest.mark.parametrize(
        "registered_user, from_user",
        [
            ("alice%40crm.example.test/app", "alice%40crm.example.test/app@sip-proxy.example.test"),
            ("alice:crm.example.test~app", "alice:crm.example.test~app@sip-proxy.example.test"),
        ],
    )
    def test_registered_client_invite_supports_old_proxy_login_syntax(
            self, deployed_crm, monkeypatch, registered_user, from_user):
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

        reg = sip_stack._Registration(
            sip_user=registered_user,
            contact_uri="sip:alice@10.0.0.5:5060",
            expires=10**12,
            user_id=42,
            subscriber_id=SUBSCRIBER_ID,
            base_url=deployed_crm.BASE_URL,
            source_addr=("10.0.0.5", 5060),
        )
        sip_stack._registrations[registered_user] = {
            "10.0.0.5:5060|sip:alice@10.0.0.5:5060": reg
        }

        invite = _invite_message(from_user, "+4930123456")
        sip_stack._handle_inbound_invite(invite, ("10.0.0.5", 5060))

        assert events, (
            "old proxy-style SIP identity did not resolve to the registered "
            "subscriber; source registration lookup regressed"
        )
        sub, key, payload = events[0]
        assert sub["id"] == SUBSCRIBER_ID
        assert key == "device_dial"
        assert payload["number"] == "+4930123456"
        assert payload["sip_user"] == registered_user
        assert any("SIP/2.0 183 Session Progress" in data for data, _ in sent)
        assert not any("SIP/2.0 404 Not Found" in data for data, _ in sent)
