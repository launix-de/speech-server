"""Tests for CallPipeExecutor DSL wiring.

Verifies that DSL strings are parsed and wired into correct Stage chains:
- sip:LEG -> call:CALL -> sip:LEG  (bidirectional bridge)
- play:X{url,loop} -> call:CALL    (source-only into mixer)
- tee sidechain wiring
- stage kill/cleanup
"""
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from speech_pipeline.telephony.pipe_executor import parse_dsl, CallPipeExecutor
from speech_pipeline.telephony.leg import Leg
from speech_pipeline.telephony import leg as leg_mod, call_state


# ---------------------------------------------------------------------------
# DSL Parser tests
# ---------------------------------------------------------------------------

class TestParseDsl:

    def test_simple_pipe(self):
        result = parse_dsl("sip:leg1 -> call:call1 -> sip:leg1")
        assert len(result) == 3
        assert result[0] == ("sip", "leg1", {})
        assert result[1] == ("call", "call1", {})
        assert result[2] == ("sip", "leg1", {})

    def test_pipe_separator(self):
        result = parse_dsl("sip:leg1 | call:call1 | sip:leg1")
        assert len(result) == 3

    def test_json_params(self):
        result = parse_dsl('play:wait{"url":"https://x.com/a.mp3","loop":true} -> call:c1')
        assert len(result) == 2
        assert result[0][0] == "play"
        assert result[0][1] == "wait"
        assert result[0][2]["url"] == "https://x.com/a.mp3"
        assert result[0][2]["loop"] is True

    def test_sip_with_callbacks(self):
        dsl = 'sip:leg1{"completed":"/cb/done"} -> call:c1 -> sip:leg1'
        result = parse_dsl(dsl)
        assert result[0][2]["completed"] == "/cb/done"
        assert result[2][2] == {}  # second sip has no params

    def test_tee_sidechain(self):
        result = parse_dsl("tee:tap1 -> stt:de -> webhook:https://x.com/stt")
        assert len(result) == 3
        assert result[0] == ("tee", "tap1", {})
        assert result[1] == ("stt", "de", {})

    def test_empty_returns_empty(self):
        assert parse_dsl("") == []

    def test_trailing_arrow_raises(self):
        with pytest.raises(ValueError):
            parse_dsl("sip:leg1 ->")

    def test_no_id(self):
        result = parse_dsl("sip -> call:c1 -> sip")
        assert result[0][1] == ""
        assert result[2][1] == ""


# ---------------------------------------------------------------------------
# Executor wiring helpers
# ---------------------------------------------------------------------------

def _make_call(call_id="call-test-1"):
    """Create a mock Call with a real ConferenceMixer."""
    from speech_pipeline.ConferenceMixer import ConferenceMixer
    call = MagicMock()
    call.call_id = call_id
    call.subscriber_id = "test-sub"
    mixer = ConferenceMixer(call_id, sample_rate=48000, frame_ms=20)
    call.mixer = mixer
    # Start mixer
    t = threading.Thread(target=mixer.run, daemon=True)
    t.start()
    call._mixer_thread = t
    return call


def _make_leg(leg_id="leg-test-1", call_id=None):
    """Create a Leg with a mock SIP session."""
    leg = Leg(leg_id, "inbound", "+49123", "TestPBX", "test-sub")
    if call_id:
        leg.call_id = call_id

    # Mock voip_call with enough interface for pipe_executor
    mock_call = MagicMock()
    mock_call.read_audio.side_effect = Exception("stopped")
    mock_call.get_dtmf.return_value = ""
    leg.voip_call = mock_call

    # Create a proper session with connected/hungup/rx_queue
    from speech_pipeline.telephony.leg import PyVoIPCallSession
    session = PyVoIPCallSession(mock_call)
    leg.sip_session = session

    leg_mod._legs[leg_id] = leg
    return leg


def _cleanup_call(call):
    call.mixer.cancel()
    call._mixer_thread.join(timeout=2)


def _cleanup_leg(leg_id):
    leg_mod._legs.pop(leg_id, None)


# ---------------------------------------------------------------------------
# Executor: validation
# ---------------------------------------------------------------------------

class TestExecutorValidation:

    def test_rejects_two_different_sip_ids(self):
        call = _make_call()
        ex = CallPipeExecutor(call)
        results = ex.add_pipes(["sip:leg1 -> call:call-test-1 -> sip:leg2"])
        assert results[0]["ok"] is False
        assert "same leg" in results[0]["error"]
        _cleanup_call(call)

    def test_rejects_wrong_call_id(self):
        call = _make_call("call-A")
        ex = CallPipeExecutor(call)
        with pytest.raises(Exception):
            ex._execute_pipe(parse_dsl("sip:leg1 -> call:call-B -> sip:leg1"))
        _cleanup_call(call)

    def test_rejects_call_as_first(self):
        call = _make_call()
        ex = CallPipeExecutor(call)
        with pytest.raises(Exception, match="first element"):
            ex._execute_pipe(parse_dsl("call:call-test-1 -> sip:leg1"))
        _cleanup_call(call)

    def test_rejects_more_than_two_calls(self):
        """At most two occurrences of ``call:ID`` per pipe."""
        call = _make_call()
        ex = CallPipeExecutor(call)
        with pytest.raises(Exception, match="At most two call"):
            ex._execute_pipe(parse_dsl(
                "call:call-test-1 -> sip:l1 -> call:call-test-1 -> call:call-test-1"))
        _cleanup_call(call)


# ---------------------------------------------------------------------------
# Executor: SIP bridge wiring
# ---------------------------------------------------------------------------

class TestExecutorSipBridge:

    def test_bidirectional_sip_bridge_creates_stages(self):
        """sip:LEG -> call:CALL -> sip:LEG must create SIPSource, ConferenceLeg, SIPSink."""
        call = _make_call()
        leg = _make_leg("leg-bridge-1")
        ex = CallPipeExecutor(call)

        # Execute the pipe — this wires everything
        ex.add_pipes([f"sip:leg-bridge-1 -> call:{call.call_id} -> sip:leg-bridge-1"])

        # Verify bridge handle was registered
        assert f"bridge:leg-bridge-1" in ex._stages

        # Verify leg was updated
        assert leg.call_id == call.call_id
        assert leg.status == "in-progress"

        # Cleanup
        ex.shutdown()
        time.sleep(0.2)
        _cleanup_call(call)
        _cleanup_leg("leg-bridge-1")

    def test_bridge_kill_cleans_up(self):
        """kill_stage('bridge:LEG') must cancel the ConferenceLeg."""
        call = _make_call()
        leg = _make_leg("leg-kill-1")
        ex = CallPipeExecutor(call)

        ex.add_pipes([f"sip:leg-kill-1 -> call:{call.call_id} -> sip:leg-kill-1"])
        assert ex.kill_stage("bridge:leg-kill-1")
        assert "bridge:leg-kill-1" not in ex._stages

        ex.shutdown()
        time.sleep(0.2)
        _cleanup_call(call)
        _cleanup_leg("leg-kill-1")


# ---------------------------------------------------------------------------
# Executor: play source into mixer
# ---------------------------------------------------------------------------

class TestExecutorPlaySource:

    def test_play_source_registers_stage(self):
        """play:ID{url} -> call:CALL must register a play handle."""
        call = _make_call()
        ex = CallPipeExecutor(call)

        ex.add_pipes([
            f'play:{call.call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call.call_id}'
        ])

        assert f"play:{call.call_id}_wait" in ex._stages

        ex.shutdown()
        time.sleep(0.2)
        _cleanup_call(call)

    def test_kill_play_stops_source(self):
        call = _make_call()
        ex = CallPipeExecutor(call)

        ex.add_pipes([
            f'play:{call.call_id}_wait{{"url":"examples/queue.mp3","loop":true}} -> call:{call.call_id}'
        ])

        killed = ex.kill_stage(f"play:{call.call_id}_wait")
        assert killed
        assert f"play:{call.call_id}_wait" not in ex._stages

        ex.shutdown()
        time.sleep(0.2)
        _cleanup_call(call)

    def test_kill_all_play(self):
        call = _make_call()
        ex = CallPipeExecutor(call)

        ex.add_pipes([
            f'play:p1{{"url":"examples/queue.mp3"}} -> call:{call.call_id}',
            f'play:p2{{"url":"examples/queue.mp3"}} -> call:{call.call_id}',
        ])

        killed = ex.kill_all_play()
        assert killed == 2

        ex.shutdown()
        time.sleep(0.2)
        _cleanup_call(call)


# ---------------------------------------------------------------------------
# Executor: shutdown cleans everything
# ---------------------------------------------------------------------------

class TestExecutorShutdown:

    def test_shutdown_clears_all(self):
        call = _make_call()
        leg = _make_leg("leg-sd-1")
        ex = CallPipeExecutor(call)

        ex.add_pipes([
            f"sip:leg-sd-1 -> call:{call.call_id} -> sip:leg-sd-1",
            f'play:{call.call_id}_wait{{"url":"examples/queue.mp3"}} -> call:{call.call_id}',
        ])

        ex.shutdown()
        assert len(ex._stages) == 0
        assert len(ex._tees) == 0

        time.sleep(0.2)
        _cleanup_call(call)
        _cleanup_leg("leg-sd-1")
