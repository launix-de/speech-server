"""Guardrails for the CRM call-teardown contract.

These tests exist to catch the exact regression that broke production:
the CRM ended a whole call via the legacy ``DELETE /api/calls/<sid>``
path while the intended contract had already moved to
``DELETE /api/pipelines?dsl=call:<sid>``.

They verify both sides we control in this repo:

1. the real CRM source in ``fop-dev/.../speech-server/calls.fop``
2. the in-process ``FakeCrm`` used by E2E tests

Without both checks, tests can stay green while the fake and the real
CRM silently drift apart again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import ACCOUNT_ID, ACCOUNT_TOKEN, SUBSCRIBER_ID
from fake_crm import FakeCrm


CRM_CALLS_FOP = Path(
    "/home/carli/projekte/fop-dev/backends/businesslogic/"
    "telefonanlage/speech-server/calls.fop"
)


@pytest.fixture
def crm(client, admin):
    client.put(
        "/api/pbx/TestPBX",
        data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
        headers=admin,
    )
    client.put(
        f"/api/accounts/{ACCOUNT_ID}",
        data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX"}),
        headers=admin,
    )
    c = FakeCrm(client, admin_headers=admin, account_token=ACCOUNT_TOKEN)
    c.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")
    yield c


class TestRealCrmSourceUsesDslCallDelete:
    """Catch legacy call-delete usage directly in the FOP source."""

    def test_calls_fop_exists(self):
        assert CRM_CALLS_FOP.exists(), f"CRM source missing: {CRM_CALLS_FOP}"

    def test_end_call_uses_dsl_call_delete(self):
        text = CRM_CALLS_FOP.read_text(encoding="utf-8", errors="replace")

        assert "/api/pipelines?dsl=" in text, (
            "CRM calls.fop no longer contains the DSL delete path for "
            "whole-call teardown"
        )
        assert "urlencode('call:' . \\$_callData['sid'])" in text, (
            "CRM calls.fop does not build DELETE /api/pipelines?dsl=call:<sid>"
        )
        assert "/api/calls/" not in text, (
            "CRM calls.fop still contains legacy /api/calls/<sid> teardown"
        )


class TestFakeCrmMatchesRealCallDeleteContract:
    """Keep the E2E fake aligned with the real CRM contract."""

    def test_last_participant_teardown_uses_call_dsl_delete(
            self, crm, monkeypatch):
        deleted: list[str] = []
        original_delete = crm.client.delete

        def _spy_delete(path, *args, **kwargs):
            deleted.append(path)
            return original_delete(path, *args, **kwargs)

        monkeypatch.setattr(crm.client, "delete", _spy_delete)

        call_db_id = 41
        call_sid = "call-test-delete-contract"
        participant_id = 7
        crm.calls[call_db_id] = {
            "caller": "+49170",
            "direction": "inbound",
            "status": "answered",
            "sid": call_sid,
        }
        crm.participants[participant_id] = {
            "call_db_id": call_db_id,
            "sid": "leg-7",
            "status": "answered",
            "number": "+49170",
        }

        crm._on_leg_ended(call_db_id, participant_id, "completed")

        assert deleted == [f"/api/pipelines?dsl=call:{call_sid}"], (
            "FakeCrm no longer mirrors the real CRM call teardown "
            "contract. Expected exactly one DELETE /api/pipelines?dsl=call:<sid>, "
            f"got {deleted!r}"
        )

    def test_last_participant_teardown_never_uses_legacy_calls_delete(
            self, crm, monkeypatch):
        deleted: list[str] = []

        def _spy_delete(path, *args, **kwargs):
            deleted.append(path)
            class _Resp:
                status_code = 204
            return _Resp()

        monkeypatch.setattr(crm.client, "delete", _spy_delete)

        call_db_id = 42
        call_sid = "call-test-no-legacy"
        participant_id = 8
        crm.calls[call_db_id] = {
            "caller": "+49171",
            "direction": "outbound",
            "status": "answered",
            "sid": call_sid,
        }
        crm.participants[participant_id] = {
            "call_db_id": call_db_id,
            "sid": "leg-8",
            "status": "answered",
            "number": "+49171",
        }

        crm._on_leg_ended(call_db_id, participant_id, "completed")

        assert all(not path.startswith("/api/calls/") for path in deleted), (
            "FakeCrm regressed to legacy DELETE /api/calls/<sid>: "
            f"{deleted!r}"
        )
