"""Guardrails for the CRM webclient media contract.

``webclient:USER`` only creates the browser slot. After the answered
callback, the CRM must attach ``codec:<session_id>`` explicitly and let
``buildLegPipes`` add transcript sidechains just like for SIP legs.
"""
from __future__ import annotations

import re
from pathlib import Path


CRM_ROOT = Path(
    "/home/carli/projekte/fop-dev/backends/businesslogic/"
    "telefonanlage/speech-server"
)
CRM_WEBCLIENT_FOP = CRM_ROOT / "webclient.fop"
CRM_CALLS_FOP = CRM_ROOT / "calls.fop"
CRM_WEBHOOKS_FOP = CRM_ROOT / "webhooks.fop"
CRM_TRANSCRIPT_FOP = CRM_ROOT / "transcript.fop"
CRM_PARTICIPANTS_FOP = Path(
    "/home/carli/projekte/fop-dev/backends/businesslogic/"
    "telefonanlage/calls-participant.fop"
)


class TestRealCrmWebclientMediaContract:

    def test_webclient_source_does_not_embed_stt_callback(self):
        text = CRM_WEBCLIENT_FOP.read_text(encoding="utf-8", errors="replace")

        assert "'stt_callback'" not in text, (
            "CRM webclient.fop must not embed transcript wiring into "
            "webclient:USER any more"
        )

    def test_outbound_webclient_source_does_not_embed_stt_callback(self):
        text = CRM_CALLS_FOP.read_text(encoding="utf-8", errors="replace")

        assert "'stt_callback'" not in text, (
            "CRM calls.fop must not embed transcript wiring into "
            "webclient participants"
        )

    def test_webclient_source_does_not_embed_media_graph_params(self):
        webclient_text = CRM_WEBCLIENT_FOP.read_text(encoding="utf-8", errors="replace")
        calls_text = CRM_CALLS_FOP.read_text(encoding="utf-8", errors="replace")

        webclient_params = re.search(r"\$params = \[(.*?)\];", webclient_text, re.S)
        calls_params = re.search(r"\$wcParams = \[(.*?)\];", calls_text, re.S)

        assert webclient_params, "webclient.fop params block not found"
        assert calls_params, "calls.fop webclient params block not found"

        for needle in ("'pipes'", "'dsl'", "'transcript'"):
            assert needle not in webclient_params.group(1), (
                f"CRM webclient.fop must stay slot-only, found {needle}"
            )
            assert needle not in calls_params.group(1), (
                f"CRM calls.fop must stay slot-only for client:* users, found {needle}"
            )

    def test_webclient_callback_builds_codec_pipe_via_buildlegpipes(self):
        text = CRM_WEBHOOKS_FOP.read_text(encoding="utf-8", errors="replace")

        assert "\\$_codecPipe = 'codec:' . \\$legSid" in text
        assert "-> call:' . \\$_callData['sid'] . ' -> codec:' . \\$legSid" in text
        assert "self::buildAndMergePipes(\\$_codecPipe, \\$_callData['sid'], \\$callDbId, \\$participantId, \\$_number);" in text

    def test_webclient_target_uses_stable_answerer_id(self):
        text = CRM_PARTICIPANTS_FOP.read_text(encoding="utf-8", errors="replace")

        assert "'client:fop_user_' . \\$me['ID']" in text
        assert "'client:fop_user_' . \\$me['user']" not in text

    def test_webclient_callback_prefers_local_participant_sid(self):
        text = CRM_WEBHOOKS_FOP.read_text(encoding="utf-8", errors="replace")

        assert "\\$legSid = trim((string)(\\$_pData['sid'] ?? \\$legSid));" in text
        assert "explode(':', \\$legSid, 2)[1]" in text

    def test_transcript_tap_id_includes_participant_namespace(self):
        text = CRM_TRANSCRIPT_FOP.read_text(encoding="utf-8", errors="replace")

        assert "\\$_localLegId" in text
        assert "\\$_tapId = \\$_localLegId . '_p' . intval(\\$participantId) . '_tap';" in text
