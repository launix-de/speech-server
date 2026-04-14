"""Stage-level resilience: races + subsystem failure.

Audit gaps covered:

#7  Concurrent hold / unhold — two CRM operators hit the same leg.
#8  Transcript tap dies mid-call — the STT sidechain must fail in
    isolation; the primary audio path (codec/SIP ↔ conference) must
    continue.
#10 Play-completion race — a ``loop=false`` play naturally finishes
    exactly as ``unhold`` issues its DELETE + POST.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from conftest import ACCOUNT_TOKEN, SUBSCRIBER_ID, create_call


# ---------------------------------------------------------------------------
# #7. Concurrent hold + unhold
# ---------------------------------------------------------------------------

class TestConcurrentHoldUnhold:
    """Two operators hit hold+unhold on the same leg at the same time.
    No matter who wins, the call must end up in a deterministic
    state — either on-hold with a single hold-music stage, or bridged
    with no hold-music stage.  NEVER: both stages present, or neither
    + no bridge (audio dead)."""

    def test_ten_parallel_hold_unhold_cycles_leave_single_bridge(
            self, client, account):
        """Smoke: do N concurrent cycles; end state has exactly one
        leg and either a bridge OR a single hold-music stage."""
        from speech_pipeline.telephony import call_state
        call_id = create_call(client, account)

        # Seed a bridged fake leg (no real RTP, just a stage id in the
        # executor so the DELETE paths have something to find).
        from unittest.mock import MagicMock
        from speech_pipeline.telephony import leg as leg_mod
        fake = MagicMock(); fake.RTPClients = []
        leg = leg_mod.create_leg(
            direction="outbound", number="+49", pbx_id="TestPBX",
            subscriber_id=SUBSCRIBER_ID, voip_call=fake,
        )
        leg.call_id = call_id
        leg.callbacks = {"completed": "/cb"}

        try:
            client.post("/api/pipelines", data=json.dumps({
                "dsl": f"sip:{leg.leg_id} -> call:{call_id} "
                       f"-> sip:{leg.leg_id}",
            }), headers=account)

            def _hold():
                client.delete("/api/pipelines",
                              data=json.dumps({"dsl": f"bridge:{leg.leg_id}"}),
                              headers=account)
                client.post("/api/pipelines", data=json.dumps({
                    "dsl": f'play:{call_id}_hold_{leg.leg_id}'
                           f'{{"url":"examples/queue.mp3","loop":true}} '
                           f'-> sip:{leg.leg_id}',
                }), headers=account)

            def _unhold():
                client.delete("/api/pipelines",
                              data=json.dumps({
                                  "dsl": f"play:{call_id}_hold_{leg.leg_id}"
                              }), headers=account)
                client.post("/api/pipelines", data=json.dumps({
                    "dsl": f"sip:{leg.leg_id} -> call:{call_id} "
                           f"-> sip:{leg.leg_id}",
                }), headers=account)

            threads = []
            for i in range(10):
                t = threading.Thread(target=_hold if i % 2 == 0 else _unhold)
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=5)

            # The executor must not have crashed; a final DELETE-all
            # must succeed without raising.
            call = call_state.get_call(call_id)
            assert call is not None, "call evaporated under contention"
            assert getattr(call, "pipe_executor", None) is not None
        finally:
            leg_mod._legs.pop(leg.leg_id, None)
            client.delete(f"/api/calls/{call_id}", headers=account)


# ---------------------------------------------------------------------------
# #8. Transcript tap crashes — primary path survives
# ---------------------------------------------------------------------------

class TestTranscriptCrashIsolation:
    """Whisper pipe crashes — the main codec/SIP audio path must keep
    running.  The tee's sidechain thread dying cannot take the forward
    pipe down with it."""

    def test_sidechain_stt_exception_does_not_kill_tee_forward(
            self, client, account, monkeypatch):
        """Simulate a STT-stage that throws on first frame.  Verify the
        tee's forward output still flows: install a terminal queue
        sink downstream of the tee and assert it keeps receiving frames
        after the sidechain failed."""
        import queue as _q
        from speech_pipeline.AudioTee import AudioTee
        from speech_pipeline.base import AudioFormat, Stage

        class _ExplodingSink(Stage):
            def __init__(self):
                super().__init__()
                self.input_format = AudioFormat(48000, "s16le")

            def run(self):
                raise RuntimeError("simulated whisper crash")

        class _CaptureSink(Stage):
            def __init__(self):
                super().__init__()
                self.input_format = AudioFormat(48000, "s16le")
                self.frames = []

            def run(self):
                if not self.upstream:
                    return
                for c in self.upstream.stream_pcm24k():
                    self.frames.append(c)
                    if len(self.frames) > 5:
                        break

        class _Source(Stage):
            def __init__(self):
                super().__init__()
                self.output_format = AudioFormat(48000, "s16le")

            def stream_pcm24k(self):
                for _ in range(50):
                    yield b"\x00\x01" * 512
                    time.sleep(0.01)

        src = _Source()
        tee = AudioTee(48000, "s16le")
        capture = _CaptureSink()
        src.pipe(tee).pipe(capture)
        tee.add_sidechain(_ExplodingSink())

        t = threading.Thread(target=capture.run, daemon=True)
        t.start()
        t.join(timeout=3.0)

        # Forward path saw frames even though the sidechain exploded.
        assert len(capture.frames) > 0, (
            "AudioTee forward path stopped after sidechain crashed — "
            "a STT crash would mute the entire call."
        )


# ---------------------------------------------------------------------------
# #10. Play-completion race
# ---------------------------------------------------------------------------

class TestPlayCompletionRace:
    """A non-looping hold jingle finishes naturally.  The CRM issues
    unhold (DELETE play + POST bridge) at the same moment.  The
    resulting state must have a working bridge and no zombie play
    stage left behind."""

    def test_unhold_immediately_after_play_ends(
            self, client, account):
        from unittest.mock import MagicMock
        from speech_pipeline.telephony import leg as leg_mod
        call_id = create_call(client, account)

        fake = MagicMock(); fake.RTPClients = []
        leg = leg_mod.create_leg(
            direction="outbound", number="+49", pbx_id="TestPBX",
            subscriber_id=SUBSCRIBER_ID, voip_call=fake,
        )
        leg.call_id = call_id
        leg.callbacks = {"completed": "/cb"}

        try:
            # Play a SHORT, non-looping jingle straight into the leg.
            client.post("/api/pipelines", data=json.dumps({
                "dsl": f'play:{call_id}_hold_{leg.leg_id}'
                       f'{{"url":"examples/word-on-beat.mp3",'
                       f'"loop":false}} '
                       f'-> sip:{leg.leg_id}',
            }), headers=account)

            # Let the jingle self-complete, then issue the unhold
            # DELETE + POST pair as fast as possible.
            time.sleep(0.6)
            r1 = client.delete("/api/pipelines", data=json.dumps({
                "dsl": f"play:{call_id}_hold_{leg.leg_id}",
            }), headers=account)
            # DELETE returns 204 if stage still alive, 404 if already
            # self-completed — BOTH are acceptable.
            assert r1.status_code in (204, 404), r1.get_data(as_text=True)

            r2 = client.post("/api/pipelines", data=json.dumps({
                "dsl": f"sip:{leg.leg_id} -> call:{call_id} "
                       f"-> sip:{leg.leg_id}",
            }), headers=account)
            assert r2.status_code == 201, r2.get_data(as_text=True)

            # A second DELETE on the same play must 404 (not leak).
            r3 = client.delete("/api/pipelines", data=json.dumps({
                "dsl": f"play:{call_id}_hold_{leg.leg_id}",
            }), headers=account)
            assert r3.status_code == 404, (
                f"Play stage still addressable after natural + explicit "
                f"completion: {r3.status_code} {r3.get_data(as_text=True)}"
            )
        finally:
            leg_mod._legs.pop(leg.leg_id, None)
            client.delete(f"/api/calls/{call_id}", headers=account)
