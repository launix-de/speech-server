from __future__ import annotations

import threading
import time

from speech_pipeline.telephony import sip_listener


class _DummySession:
    def __init__(self):
        self.hungup = threading.Event()


class _DummyLeg:
    def __init__(self, leg_id: str):
        self.leg_id = leg_id
        self.call_id = None
        self.status = "ringing"
        self.sip_session = _DummySession()


def test_wait_for_bridge_allows_late_bridge_within_grace(monkeypatch):
    leg = _DummyLeg("leg-late-bridge")
    deleted: list[str] = []

    monkeypatch.setattr(sip_listener, "RING_TIMEOUT", 0.1)
    monkeypatch.setattr(sip_listener, "LATE_BRIDGE_GRACE_TIMEOUT", 0.65)
    monkeypatch.setattr(
        sip_listener.leg_mod,
        "delete_leg",
        lambda leg_id: deleted.append(leg_id) or True,
    )

    def _bridge_late():
        time.sleep(0.2)
        leg.call_id = "call-late"

    threading.Thread(target=_bridge_late, daemon=True).start()

    sip_listener._wait_for_bridge(leg, None)

    assert leg.call_id == "call-late"
    assert deleted == [], "late bridge during grace must keep the SIP leg alive"


def test_wait_for_bridge_hangs_up_after_grace_expires(monkeypatch):
    leg = _DummyLeg("leg-hard-timeout")
    deleted: list[str] = []

    monkeypatch.setattr(sip_listener, "RING_TIMEOUT", 0.1)
    monkeypatch.setattr(sip_listener, "LATE_BRIDGE_GRACE_TIMEOUT", 0.55)
    monkeypatch.setattr(
        sip_listener.leg_mod,
        "delete_leg",
        lambda leg_id: deleted.append(leg_id) or True,
    )

    sip_listener._wait_for_bridge(leg, None)

    assert leg.status == "no-answer"
    assert deleted == ["leg-hard-timeout"]


def test_wait_for_bridge_without_timeout_waits_for_remote_hangup(monkeypatch):
    leg = _DummyLeg("leg-no-timeout")
    deleted: list[str] = []

    monkeypatch.setattr(
        sip_listener.leg_mod,
        "delete_leg",
        lambda leg_id: deleted.append(leg_id) or True,
    )

    def _hangup_late():
        time.sleep(0.2)
        leg.sip_session.hungup.set()

    threading.Thread(target=_hangup_late, daemon=True).start()

    sip_listener._wait_for_bridge(leg, None, ring_timeout=-1)

    assert leg.status == "completed"
    assert deleted == ["leg-no-timeout"]
