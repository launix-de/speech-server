"""Unit tests for the telephony DSL validation rules (_validate_elements).

Tests validation in isolation using a mock call — no Flask app needed.
"""
import pytest
from speech_pipeline.telephony.pipe_executor import CallPipeExecutor


def _make_executor(call_id="call-test123"):
    """Create a minimal CallPipeExecutor with a mock call."""
    call = type("MockCall", (), {
        "call_id": call_id,
        "mixer": type("MockMixer", (), {"sample_rate": 48000})(),
    })()
    return CallPipeExecutor(call)


# ===========================================================================
# Empty / basic
# ===========================================================================

class TestEmptyPipe:
    def test_empty_pipe_rejected(self):
        ex = _make_executor()
        with pytest.raises(ValueError, match="Empty pipe"):
            ex._validate_elements([])


# ===========================================================================
# call rules
# ===========================================================================

class TestCallStageRules:
    def test_multiple_call_stages_rejected(self):
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("call", "call-test123", {}),
                     ("call", "call-test123", {})]
        with pytest.raises(ValueError, match="Only one call"):
            ex._validate_elements(elements)

    def test_call_as_first_element_rejected(self):
        ex = _make_executor()
        elements = [("call", "call-test123", {}), ("sip", "l1", {})]
        with pytest.raises(ValueError, match="cannot be the first"):
            ex._validate_elements(elements)

    def test_call_id_mismatch_rejected(self):
        ex = _make_executor()
        elements = [("play", "x", {}), ("call", "call-wrong", {})]
        with pytest.raises(ValueError, match="executor is bound to"):
            ex._validate_elements(elements)

    def test_call_missing_id_rejected(self):
        ex = _make_executor()
        elements = [("play", "x", {}), ("call", "", {})]
        with pytest.raises(ValueError, match="call requires a call id"):
            ex._validate_elements(elements)

    def test_call_followed_by_non_sip_rejected(self):
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("call", "call-test123", {}),
                     ("webhook", "https://example.com", {})]
        with pytest.raises(ValueError, match="call may only be followed"):
            ex._validate_elements(elements)

    def test_call_in_middle_valid(self):
        """sip -> call -> sip is the canonical bidirectional pattern."""
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("call", "call-test123", {}),
                     ("sip", "l1", {})]
        ex._validate_elements(elements)  # should not raise

    def test_call_at_end_valid(self):
        """play -> call is valid (source into conference)."""
        ex = _make_executor()
        elements = [("play", "x", {}), ("call", "call-test123", {})]
        ex._validate_elements(elements)

    def test_single_sip_after_call_valid(self):
        """sip -> call -> sip (bidirectional bridge)."""
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("call", "call-test123", {}),
                     ("sip", "l1", {})]
        ex._validate_elements(elements)


# ===========================================================================
# sip rules
# ===========================================================================

class TestSipStageRules:
    def test_three_sip_stages_rejected(self):
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("call", "call-test123", {}),
                     ("sip", "l1", {}), ("sip", "l1", {})]
        with pytest.raises(ValueError, match="At most two sip"):
            ex._validate_elements(elements)

    def test_bidirectional_sip_different_legs_rejected(self):
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("call", "call-test123", {}),
                     ("sip", "l2", {})]
        with pytest.raises(ValueError, match="same leg"):
            ex._validate_elements(elements)

    def test_sip_without_call_not_terminal_rejected(self):
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("stt", "de", {})]
        with pytest.raises(ValueError, match="Without call.*terminal sink"):
            ex._validate_elements(elements)

    def test_sip_as_sole_element_rejected(self):
        ex = _make_executor()
        elements = [("sip", "l1", {})]
        with pytest.raises(ValueError, match="upstream source"):
            ex._validate_elements(elements)

    def test_sip_to_sip_without_call_rejected(self):
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("sip", "l1", {})]
        with pytest.raises(ValueError, match="Without call.*terminal sink"):
            ex._validate_elements(elements)

    def test_sip_as_source_then_sip_as_sink(self):
        """sip:l1 -> sip:l1 without call — not valid (needs call in between)."""
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("sip", "l1", {})]
        with pytest.raises(ValueError):
            ex._validate_elements(elements)

    def test_source_to_sip_terminal_valid(self):
        """tts -> sip:l1 (no call, sip as terminal sink)."""
        ex = _make_executor()
        elements = [("tts", "de", {}), ("sip", "l1", {})]
        ex._validate_elements(elements)

    def test_sip_source_to_sip_sink_rejected_without_call(self):
        """Direct sip -> sip requires source in front."""
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("sip", "l2", {})]
        with pytest.raises(ValueError):
            ex._validate_elements(elements)


# ===========================================================================
# webhook rules
# ===========================================================================

class TestWebhookStageRules:
    def test_webhook_not_terminal_rejected(self):
        ex = _make_executor()
        elements = [("webhook", "https://example.com", {}),
                     ("call", "call-test123", {})]
        with pytest.raises(ValueError, match="webhook must be the final"):
            ex._validate_elements(elements)

    def test_webhook_as_terminal_valid(self):
        ex = _make_executor()
        elements = [("stt", "de", {}), ("webhook", "https://example.com", {})]
        ex._validate_elements(elements)

    def test_webhook_in_middle_rejected(self):
        ex = _make_executor()
        elements = [("stt", "de", {}),
                     ("webhook", "https://example.com", {}),
                     ("stt", "en", {})]
        with pytest.raises(ValueError, match="webhook must be the final"):
            ex._validate_elements(elements)


# ===========================================================================
# tee sidechain rules
# ===========================================================================

class TestTeeSidechainRules:
    def test_tee_as_first_without_existing_rejected(self):
        ex = _make_executor()
        elements = [("tee", "new-tee", {}), ("stt", "de", {})]
        with pytest.raises(ValueError, match="tee may only start.*existing"):
            ex._validate_elements(elements)

    def test_tee_sidechain_forbidden_call(self):
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {}), ("call", "x", {})]
        with pytest.raises(ValueError, match="call is not allowed"):
            ex._validate_elements(elements)

    def test_tee_sidechain_forbidden_sip(self):
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {}), ("sip", "x", {})]
        with pytest.raises(ValueError, match="sip is not allowed"):
            ex._validate_elements(elements)

    def test_tee_sidechain_forbidden_play(self):
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {}), ("play", "x", {})]
        with pytest.raises(ValueError, match="play is not allowed"):
            ex._validate_elements(elements)

    def test_tee_sidechain_forbidden_tts(self):
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {}), ("tts", "x", {})]
        with pytest.raises(ValueError, match="tts is not allowed"):
            ex._validate_elements(elements)

    def test_tee_sidechain_too_short(self):
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {})]
        with pytest.raises(ValueError, match="at least one downstream"):
            ex._validate_elements(elements)

    def test_tee_sidechain_stt_allowed(self):
        """STT is allowed in sidechains."""
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {}), ("stt", "de", {})]
        ex._validate_elements(elements)

    def test_tee_sidechain_webhook_allowed(self):
        """webhook is allowed in sidechains."""
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {}),
                     ("stt", "de", {}),
                     ("webhook", "https://example.com", {})]
        ex._validate_elements(elements)

    def test_tee_sidechain_multi_stage_valid(self):
        """Multi-stage sidechain: tee -> stt -> webhook."""
        from speech_pipeline.AudioTee import AudioTee
        ex = _make_executor()
        ex._tees["existing"] = AudioTee(48000, "s16le")
        elements = [("tee", "existing", {}),
                     ("stt", "de", {}),
                     ("webhook", "https://example.com", {})]
        ex._validate_elements(elements)


# ===========================================================================
# Valid patterns (positive tests)
# ===========================================================================

class TestValidPatterns:
    def test_play_to_call(self):
        ex = _make_executor()
        elements = [("play", "x", {}), ("call", "call-test123", {})]
        ex._validate_elements(elements)

    def test_tts_to_call(self):
        ex = _make_executor()
        elements = [("tts", "de", {}), ("call", "call-test123", {})]
        ex._validate_elements(elements)

    def test_bidirectional_sip(self):
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("call", "call-test123", {}),
                     ("sip", "l1", {})]
        ex._validate_elements(elements)

    def test_stt_to_webhook(self):
        ex = _make_executor()
        elements = [("stt", "de", {}), ("webhook", "https://example.com", {})]
        ex._validate_elements(elements)

    def test_source_to_sip_terminal(self):
        ex = _make_executor()
        elements = [("tts", "de", {}), ("sip", "l1", {})]
        ex._validate_elements(elements)

    def test_play_to_sip_terminal(self):
        ex = _make_executor()
        elements = [("play", "x", {}), ("sip", "l1", {})]
        ex._validate_elements(elements)

    def test_sip_tee_call_sip(self):
        """sip -> tee -> call -> sip (bridge with tee for recording sidechain)."""
        ex = _make_executor()
        elements = [("sip", "l1", {}), ("tee", "t1", {}),
                     ("call", "call-test123", {}), ("sip", "l1", {})]
        ex._validate_elements(elements)
