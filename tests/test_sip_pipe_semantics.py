"""Neighbour-based sip:LEG piping + Stage.close() lifecycle.

Covers the invariant: *two items next to each other in DSL are always
piped into each other.*  SIP wrappers (SIPSource / SIPSink) are created
based on the existence of left/right neighbours.

Also covers the orderly teardown API:

* ``Stage.close()`` propagates both directions, idempotent via the
  ``closed`` flag, and invokes ``_on_close()`` exactly once per stage.
* ``SIPSource._on_close`` / ``SIPSink._on_close`` send SIP BYE/CANCEL
  (idempotent against ``session.hungup``).
* ``HangupSink.run()`` calls ``self.upstream.close()`` — nothing else.
* ``ConferenceMixer`` auto-removes drained sources so idle cleanup
  can fire for non-loop plays.

End-to-end drain cascade for ``play:X -> call:C``:
    play yields last chunk → QueueSink EOF → mixer source drains →
    auto-removed from ``_sources`` → ``IDLE_TIMEOUT`` fires →
    ``_auto_cleanup`` → ``delete_leg`` → BYE.
"""
from __future__ import annotations

import queue as _queue
import threading
import time
from unittest.mock import MagicMock

import pytest

from speech_pipeline.base import Stage
from speech_pipeline.ConferenceMixer import ConferenceMixer
from speech_pipeline.HangupSink import HangupSink
from speech_pipeline.SIPSink import SIPSink
from speech_pipeline.SIPSource import SIPSource
from speech_pipeline.telephony import leg as leg_mod
from speech_pipeline.telephony.pipe_executor import CallPipeExecutor


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal stand-in for PyVoIPCallSession / RTPSession."""
    def __init__(self):
        self.connected = threading.Event()
        self.connected.set()
        self.hungup = threading.Event()
        self.rx_queue = _queue.Queue()
        self.call = MagicMock(spec=[])
        self.hangup = MagicMock(side_effect=lambda: self.hungup.set())


def _make_leg(leg_id: str) -> leg_mod.Leg:
    leg = leg_mod.Leg(
        leg_id=leg_id, direction="inbound", number="+49123",
        pbx_id="pbx", subscriber_id="sub",
    )
    leg.voip_call = MagicMock()
    leg.sip_session = _FakeSession()
    leg.hangup = MagicMock(side_effect=leg.sip_session.hangup)
    leg_mod._legs[leg_id] = leg
    return leg


@pytest.fixture
def leg_l():
    leg = _make_leg("test-l")
    yield leg
    leg_mod._legs.pop("test-l", None)


@pytest.fixture
def leg_a():
    leg = _make_leg("test-a")
    yield leg
    leg_mod._legs.pop("test-a", None)


# ===========================================================================
# Stage.close() base API
# ===========================================================================

class TestStageClose:
    """The orderly-teardown API on Stage itself."""

    def test_close_invokes_on_close_once(self):
        calls = []
        class S(Stage):
            def _on_close(self): calls.append(1)
        s = S()
        s.close()
        s.close()
        assert calls == [1]
        assert s.closed is True

    def test_close_propagates_downstream(self):
        class S(Stage):
            hook = 0
            def _on_close(self): type(self).hook += 1
        a, b = S(), S()
        a.pipe(b)
        a.close()
        assert a.closed and b.closed
        assert S.hook == 2

    def test_close_propagates_upstream(self):
        """HangupSink-style: downstream-initiated close reaches upstream."""
        calls = []
        class S(Stage):
            def __init__(self, name):
                super().__init__(); self.name = name
            def _on_close(self): calls.append(self.name)
        a = S("a"); b = S("b"); c = S("c")
        a.pipe(b).pipe(c)
        c.close()
        assert calls == ["c", "b", "a"]

    def test_close_idempotent_in_cycle_like_chains(self):
        """Propagating both ways must not loop indefinitely."""
        class S(Stage):
            pass
        a, b = S(), S()
        a.pipe(b)
        a.close()  # goes right to b, b tries to go back to a — guarded
        b.close()  # no-op
        assert a.closed and b.closed


# ===========================================================================
# SIP wrapper _on_close hooks
# ===========================================================================

class TestSipOnClose:

    def test_sip_source_on_close_calls_leg_hangup(self, leg_l):
        src = SIPSource(leg_l.sip_session, leg=leg_l)
        src.close()
        leg_l.hangup.assert_called_once()

    def test_sip_sink_on_close_calls_leg_hangup(self, leg_l):
        sink = SIPSink(leg_l.sip_session, leg=leg_l)
        sink.close()
        leg_l.hangup.assert_called_once()

    def test_on_close_idempotent_when_already_hungup(self, leg_l):
        leg_l.sip_session.hungup.set()
        src = SIPSource(leg_l.sip_session, leg=leg_l)
        src.close()
        leg_l.hangup.assert_not_called()

    def test_on_close_falls_back_to_session_hangup(self):
        """No leg passed → session.hangup() is used directly."""
        sess = _FakeSession()
        src = SIPSource(sess, leg=None)
        src.close()
        sess.hangup.assert_called_once()


# ===========================================================================
# HangupSink behaviour
# ===========================================================================

class TestHangupSink:

    def test_run_closes_upstream(self):
        hs = HangupSink()
        up = MagicMock(spec=Stage)
        up.closed = False
        hs.upstream = up
        hs.run()
        up.close.assert_called_once()

    def test_run_without_upstream_does_nothing(self):
        hs = HangupSink()
        hs.run()  # must not raise

    def test_sip_to_hangup_chain_fires_leg_hangup(self, leg_l):
        """Integration: SIPSource -> HangupSink. HangupSink.run() closes
        upstream; close propagates; SIPSource._on_close hangs the leg."""
        src = SIPSource(leg_l.sip_session, leg=leg_l)
        hs = HangupSink()
        src.pipe(hs)
        hs.run()
        leg_l.hangup.assert_called_once()


# ===========================================================================
# SIPSink natural-EOF → close
# ===========================================================================

class TestSipSinkEofClose:

    def test_natural_eof_calls_close(self, leg_l):
        """Upstream generator exhausts → SIPSink.close() → leg.hangup."""
        sink = SIPSink(leg_l.sip_session, leg=leg_l)

        class _EOFSource:
            output_format = None
            closed = False
            upstream = None
            def stream_pcm24k(self):
                return iter((b"\x00\x00" * 160,))  # one chunk then EOF
            def close(self): self.closed = True

        src = _EOFSource()
        sink.upstream = src
        # Force pyVoIP path, one RTP client stub to reach the loop.
        # Session.call must be a plain MagicMock (no spec) so write_audio
        # exists.
        sink.session.call = MagicMock()
        rtp = MagicMock()
        rtp.preference = MagicMock()
        sink.session.call.RTPClients = [rtp]
        sink.run()
        assert sink.closed is True
        leg_l.hangup.assert_called_once()

    def test_no_rtp_clients_skips_close(self, leg_l):
        """pyVoIP early-return (no RTPClients) must NOT trigger hangup."""
        sink = SIPSink(leg_l.sip_session, leg=leg_l)

        class _DummySource:
            output_format = None
            def stream_pcm24k(self): return iter(())
            def close(self): pass

        sink.upstream = _DummySource()
        sink.session.call.RTPClients = []
        sink.run()
        assert sink.closed is False
        leg_l.hangup.assert_not_called()

    def test_cancelled_mid_stream_skips_close(self, leg_l):
        """External cancel during the write loop must NOT trigger hangup."""
        sink = SIPSink(leg_l.sip_session, leg=leg_l)

        def _infinite():
            while True:
                yield b"\x00\x00" * 160

        class _Source:
            output_format = None
            def stream_pcm24k(self): return _infinite()
            def close(self): pass

        sink.upstream = _Source()
        sink.session.call = MagicMock()
        rtp = MagicMock()
        rtp.preference = MagicMock()
        sink.session.call.RTPClients = [rtp]

        def _cancel_soon():
            time.sleep(0.05)
            sink.cancelled = True
        threading.Thread(target=_cancel_soon, daemon=True).start()

        sink.run()
        assert sink.closed is False
        leg_l.hangup.assert_not_called()


# ===========================================================================
# _wrap_sip structural tests
# ===========================================================================

class TestWrapSipStructure:

    def _resolve(self, ex, elements):
        return [(t, i, p, ex._create_stage(t, i, p)) for t, i, p in elements]

    def test_sip_hangup_single_chain(self, leg_l):
        ex = CallPipeExecutor()
        resolved = self._resolve(ex, [("sip", "test-l", {}), ("hangup", "", {})])
        chains = ex._wrap_sip(resolved)
        assert len(chains) == 1
        assert chains[0][0][0] == "sip_source"
        assert isinstance(chains[0][1][3], HangupSink)

    def test_play_sip_terminal(self, leg_l):
        ex = CallPipeExecutor()
        resolved = self._resolve(ex, [
            ("play", "test-l", {"url": "http://example.com/a.wav"}),
            ("sip", "test-l", {}),
        ])
        chains = ex._wrap_sip(resolved)
        assert len(chains) == 1
        assert chains[0][-1][0] == "sip_sink"

    def test_middle_sip_splits_into_two_chains(self, leg_l):
        ex = CallPipeExecutor()
        resolved = self._resolve(ex, [
            ("play", "test-l", {"url": "http://example.com/a.wav"}),
            ("sip", "test-l", {}),
            ("save", "out", {}),
        ])
        chains = ex._wrap_sip(resolved)
        assert [t[0] for t in chains[0]] == ["play", "sip_sink"]
        assert [t[0] for t in chains[1]] == ["sip_source", "save"]
        # Both wrappers share the same session (enforced shared leg state)
        assert chains[0][1][3].session is chains[1][0][3].session

    def test_bridge_same_leg_twice_one_chain(self, leg_a):
        ex = CallPipeExecutor()
        sess = leg_a.sip_session
        resolved = [
            ("sip", "test-a", {}, sess),
            ("call", "c1", {}, object()),
            ("sip", "test-a", {}, sess),
        ]
        chains = ex._wrap_sip(resolved)
        assert len(chains) == 1
        assert [t[0] for t in chains[0]] == ["sip_source", "call", "sip_sink"]

    def test_solo_sip_rejected(self, leg_l):
        ex = CallPipeExecutor()
        resolved = self._resolve(ex, [("sip", "test-l", {})])
        with pytest.raises(ValueError, match="standalone has no neighbours"):
            ex._wrap_sip(resolved)


# ===========================================================================
# ConferenceMixer: drained source auto-removal
# ===========================================================================

class TestMixerAutoRemoveDrainedSource:
    """The drain cascade for `play:X -> call:C` hinges on the mixer
    removing finished sources so idle cleanup can eventually fire."""

    def test_source_removed_after_drain(self):
        m = ConferenceMixer("test", sample_rate=48000, frame_samples=480)
        m.IDLE_TIMEOUT_SECONDS = 100.0  # don't actually cancel in this test

        q = m.add_input()
        # Feed one short chunk + EOF.
        q.put(b"\x00\x00" * 100)
        q.put(None)

        t = threading.Thread(target=m.run, daemon=True)
        t.start()
        try:
            # Wait until the mixer has drained the source.
            deadline = time.time() + 3
            while time.time() < deadline:
                with m._lock:
                    if not m._sources:
                        break
                time.sleep(0.02)
            with m._lock:
                assert not m._sources, "drained source should have been removed"
        finally:
            m.cancel()

    def test_idle_cleanup_fires_after_drain(self):
        m = ConferenceMixer("test2", sample_rate=48000, frame_samples=480)
        m.IDLE_TIMEOUT_SECONDS = 0.3
        fired = threading.Event()
        m.on_idle_cancel = lambda _m: fired.set()

        q = m.add_input()
        q.put(b"\x00\x00" * 100)
        q.put(None)

        t = threading.Thread(target=m.run, daemon=True)
        t.start()
        try:
            assert fired.wait(timeout=3.0), "idle cleanup did not fire after drain"
        finally:
            m.cancel()


# ===========================================================================
# End-to-end: sip:L -> hangup (no call context)
# ===========================================================================

class TestDslEndToEnd:

    def test_sip_to_hangup_closes_leg(self, leg_l):
        ex = CallPipeExecutor()
        ex.add_pipes(["sip:test-l -> hangup"])
        for _ in range(100):
            if leg_l.hangup.called:
                break
            time.sleep(0.02)
        leg_l.hangup.assert_called()


# ===========================================================================
# call:C coupling — `sip -> call -> sip` ≡ `call -> sip -> call`
# ===========================================================================

class TestCallCouplingSemantics:
    """Two occurrences of the same ``call:ID`` denote input+output of
    the same virtual leg.  Invariant test — see pipe_executor docstring."""

    def test_two_call_positions_share_coupling(self, leg_a):
        """``call:C -> sip:A -> call:C`` resolves to ConferenceSource +
        ConferenceSink sharing one ``_Coupling`` (= same virtual leg)."""
        from speech_pipeline.ConferenceEndpoint import (
            ConferenceSource, ConferenceSink,
        )
        from speech_pipeline.telephony import call_state

        # Create a real call so the coupling has a mixer to register on
        call = call_state.create_call("sub", "acc", "pbx")
        try:
            ex = CallPipeExecutor(call=call)
            resolved = [(t, i, p, ex._create_stage(t, i, p)) for t, i, p in
                        [("call", call.call_id, {}),
                         ("sip", "test-a", {}),
                         ("call", call.call_id, {})]]
            wrapped = ex._wrap_calls(resolved)
            types = [w[0] for w in wrapped]
            assert types == ["call_source", "sip", "call_sink"]
            assert isinstance(wrapped[0][3], ConferenceSource)
            assert isinstance(wrapped[2][3], ConferenceSink)
            # Shared coupling → same mix-minus leg
            assert wrapped[0][3].coupling is wrapped[2][3].coupling
        finally:
            call_state.delete_call(call.call_id)

    def test_call_call_different_ids_rejected(self):
        """Two call stages with different ids → validation error."""
        from speech_pipeline.telephony import call_state
        call = call_state.create_call("sub", "acc", "pbx")
        try:
            ex = CallPipeExecutor(call=call)
            elems = [("call", call.call_id, {}),
                     ("sip", "x", {}),
                     ("call", "other", {})]
            with pytest.raises(ValueError, match="same call id"):
                ex._validate_elements(elems)
        finally:
            call_state.delete_call(call.call_id)

    def test_three_call_positions_rejected(self):
        from speech_pipeline.telephony import call_state
        call = call_state.create_call("sub", "acc", "pbx")
        try:
            ex = CallPipeExecutor(call=call)
            elems = [("call", call.call_id, {}),
                     ("sip", "x", {}),
                     ("call", call.call_id, {}),
                     ("call", call.call_id, {})]
            with pytest.raises(ValueError, match="At most two call"):
                ex._validate_elements(elems)
        finally:
            call_state.delete_call(call.call_id)


# ===========================================================================
# Audio-quality: bidirectional coupling produces the expected mix-minus
# ===========================================================================

class TestCouplingAudioQuality:
    """End-to-end: push audio through a _Coupling and verify that
    * the mixer forwards every sample we push as an input source, and
    * the output queue is mix-minus (= silence, because only our own
      input exists and is subtracted)."""

    def test_mix_minus_subtracts_own_input(self):
        """Single coupled endpoint: push known audio, expect silence out."""
        from speech_pipeline.ConferenceEndpoint import _Coupling

        mixer = ConferenceMixer("q-test", sample_rate=48000, frame_samples=480)
        mixer.IDLE_TIMEOUT_SECONDS = 60.0
        coupling = _Coupling(mixer)
        coupling.activate()

        t = threading.Thread(target=mixer.run, daemon=True)
        t.start()
        try:
            tone = b"\x10\x27" * 480  # 10000 amplitude, 20ms @ 48k
            for _ in range(10):
                coupling.in_queue.put(tone)

            collected = []
            deadline = time.time() + 2.0
            while len(collected) < 5 and time.time() < deadline:
                try:
                    frame = coupling.out_queue.get(timeout=0.3)
                except _queue.Empty:
                    continue
                if frame is None:
                    break
                collected.append(frame)

            assert len(collected) >= 3
            # Mix-minus against own input → output must be silence.
            for f in collected[1:]:  # allow first frame to have residual
                silence = b"\x00" * len(f)
                assert f == silence, "mix-minus failed: non-silent output"
        finally:
            mixer.cancel()

    def test_second_participant_hears_first(self):
        """Two coupled endpoints: each hears the other, not itself."""
        from speech_pipeline.ConferenceEndpoint import _Coupling

        mixer = ConferenceMixer("q-test2", sample_rate=48000, frame_samples=480)
        mixer.IDLE_TIMEOUT_SECONDS = 60.0
        c1 = _Coupling(mixer)
        c2 = _Coupling(mixer)
        c1.activate()
        c2.activate()

        t = threading.Thread(target=mixer.run, daemon=True)
        t.start()
        try:
            tone1 = (b"\x10\x27" * 480)   # amp 10000
            tone2 = (b"\xf0\xd8" * 480)   # amp -10000
            for _ in range(6):
                c1.in_queue.put(tone1)
                c2.in_queue.put(tone2)

            # c1 should hear tone2, c2 should hear tone1.
            def _drain(q, n):
                out = []
                for _ in range(n):
                    try:
                        f = q.get(timeout=0.3)
                    except _queue.Empty:
                        return out
                    if f is None:
                        return out
                    out.append(f)
                return out

            c1_out = _drain(c1.out_queue, 5)
            c2_out = _drain(c2.out_queue, 5)
            assert len(c1_out) >= 3 and len(c2_out) >= 3
            # Both should be non-silent (they hear the other party).
            silence = b"\x00" * len(c1_out[-1])
            assert c1_out[-1] != silence
            assert c2_out[-1] != silence
        finally:
            mixer.cancel()


# ===========================================================================
# DELETE cascade — orphaning one side hangs up the other
# ===========================================================================

class TestDeleteCascade:

    def test_delete_call_hangs_up_sip_leg(self, leg_l):
        """``DELETE /api/calls/{id}`` must hangup all SIP legs in that call."""
        from speech_pipeline.telephony import call_state
        call = call_state.create_call("sub", "acc", "pbx")
        leg_l.call_id = call.call_id
        try:
            call_state.delete_call(call.call_id)
            leg_l.hangup.assert_called()
            assert call_state.get_call(call.call_id) is None
        finally:
            call_state.delete_call(call.call_id)

    def test_delete_leg_sets_session_hungup(self, leg_l):
        """``delete_leg`` → leg.hangup() → session.hungup set.
        A ConferenceSink feeding this leg sees upstream EOF and
        closes; the mixer source drains and idle-cleanup can start."""
        leg_mod.delete_leg("test-l")
        # Our mocked leg.hangup is sticky; session.hungup gets set by it.
        assert leg_l.sip_session.hungup.is_set()
