"""A missing REGISTER response must not stop provider registration forever."""
from unittest.mock import Mock

import pytest

from speech_pipeline.telephony import sip_stack


class _Timer:
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.cancelled = False
        self.daemon = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.function()


@pytest.fixture
def trunk(monkeypatch):
    record = sip_stack._Trunk("test", "127.0.0.1", 5060, "user", "secret")
    monkeypatch.setattr(sip_stack, "_trunks", {"test": record})
    monkeypatch.setattr(sip_stack, "_transactions", {})
    monkeypatch.setattr(sip_stack, "_running", True)
    monkeypatch.setattr(sip_stack, "_public_ip", "127.0.0.1")
    monkeypatch.setattr(sip_stack.threading, "Timer", _Timer)
    monkeypatch.setattr(sip_stack, "_send", Mock())
    return record


def response(trunk, status=200, **headers):
    return {"status": status, "reason": "test", "headers": {
        "call-id": trunk.call_id_reg, "cseq": f"{trunk.cseq} REGISTER", **headers,
    }}


def test_lost_response_retries_and_keeps_retrying(trunk):
    sip_stack._send_register(trunk)
    for expected in range(2, 5):
        timer = trunk._refresh_timer
        assert timer.interval == 30
        timer.fire()
        assert trunk.cseq == expected
        assert trunk._refresh_timer is not timer
    assert sip_stack._send.call_count == 4


@pytest.mark.parametrize("expires", [20, 60, 300])
def test_success_replaces_watchdog_and_refreshes_before_expiry(trunk, expires):
    sip_stack._send_register(trunk)
    watchdog = trunk._refresh_timer
    sip_stack._handle_response(response(trunk, expires=str(expires)), (trunk.server, 5060))
    assert watchdog.cancelled
    assert trunk.registered
    assert 0 < trunk._refresh_timer.interval < expires
    assert trunk._registered_until > sip_stack.time.monotonic()
    watchdog.fire()  # a canceled timer callback that was already queued
    assert sip_stack._send.call_count == 1
    trunk._refresh_timer.fire()
    assert sip_stack._send.call_count == 2
    assert trunk._refresh_timer.interval == 30


def test_contact_expiry_takes_precedence(trunk):
    sip_stack._send_register(trunk)
    sip_stack._handle_register_response(trunk, response(
        trunk, expires="300", contact='<sip:user@127.0.0.1>;expires=20',
    ))
    assert trunk._refresh_timer.interval < 20


def test_auth_retry_also_has_response_watchdog(trunk):
    sip_stack._send_register(trunk)
    old = trunk._refresh_timer
    sip_stack._handle_response(response(trunk, 401, **{
        "www-authenticate": 'Digest realm="test", nonce="nonce", algorithm=MD5',
    }), (trunk.server, 5060))
    assert old.cancelled
    assert sip_stack._send.call_count == 2
    assert 'Authorization: Digest' in sip_stack._send.call_args.args[0]
    assert trunk._refresh_timer.interval == 30


@pytest.mark.parametrize("action", ["replace", "remove", "shutdown"])
def test_obsolete_timer_cannot_reregister_trunk(trunk, action, monkeypatch):
    sip_stack._send_register(trunk)
    timer = trunk._refresh_timer
    if action == "replace":
        sip_stack._trunks["test"] = sip_stack._Trunk("test", "127.0.0.1", 5060, "new", "secret")
    elif action == "remove":
        sip_stack._trunks.clear()
    else:
        monkeypatch.setattr(sip_stack, "_running", False)
    timer.fire()
    assert sip_stack._send.call_count == 1


def test_expired_registration_cannot_be_used_for_outbound_calls(trunk):
    trunk.registered = True
    trunk._registered_until = sip_stack.time.monotonic() - 1
    with pytest.raises(RuntimeError, match="registration expired"):
        sip_stack.call(trunk.pbx_id, "+491234")
    assert not trunk.registered


def test_rejected_registration_is_not_marked_registered(trunk):
    sip_stack._send_register(trunk)
    trunk.registered = True
    sip_stack._handle_register_response(trunk, response(trunk, 503))
    assert not trunk.registered
    assert trunk._refresh_timer.interval == 30
