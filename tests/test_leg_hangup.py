"""Tests for Leg.hangup() — ensure all SIP signaling paths are reached."""
from unittest.mock import patch, MagicMock

from speech_pipeline.telephony.leg import Leg


def _make_leg(**overrides):
    defaults = dict(
        leg_id="test-leg-1",
        direction="outbound",
        number="+491234567890",
        pbx_id="TestPBX",
        subscriber_id="test-sub",
    )
    defaults.update(overrides)
    return Leg(**defaults)


class TestTrunkHangup:
    """Trunk calls set sip_call_id (no underscore) on the leg.
    hangup() must use that attribute to call hangup_trunk_leg()."""

    def test_trunk_hangup_called_when_sip_call_id_set(self):
        leg = _make_leg()
        leg.sip_call_id = "call-id-abc"

        with patch("speech_pipeline.telephony.sip_stack.hangup_trunk_leg",
                   return_value=True) as mock:
            leg.hangup()
            mock.assert_called_once_with("call-id-abc")

    def test_trunk_hangup_skipped_when_no_sip_call_id(self):
        leg = _make_leg()
        with patch("speech_pipeline.telephony.sip_stack.hangup_trunk_leg") as mock:
            leg.hangup()
            mock.assert_not_called()


class TestSipSessionHangup:
    """When sip_session is set, hangup() must call session.hangup()."""

    def test_sip_session_hangup_called(self):
        leg = _make_leg()
        mock_session = MagicMock()
        leg.sip_session = mock_session
        leg.hangup()
        mock_session.hangup.assert_called_once()

    def test_sip_session_hangup_exception_swallowed(self):
        leg = _make_leg()
        mock_session = MagicMock()
        mock_session.hangup.side_effect = RuntimeError("boom")
        leg.sip_session = mock_session
        leg.hangup()  # must not raise
