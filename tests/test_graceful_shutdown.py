"""SIGTERM/SIGINT must drain active calls before exit.

Production deployment via pm2: a graceless shutdown leaves SIP peers
with dead RTP sockets, no BYE — they only learn the call died via
their own retransmit timeouts (10–30 s).  The graceful path issues
``delete_call`` for every active call so the peers get clean BYEs.

These tests don't actually send signals (would kill pytest); they
extract the registered handler from the ``main()`` setup and call
it directly.
"""
from __future__ import annotations

import signal as _signal
from unittest.mock import patch

import pytest


@pytest.fixture
def main_handler():
    """Run piper_multi_server.main() up to the point where it
    registers the graceful-shutdown handler, intercept it, then
    abort before app.run.

    Returns the captured shutdown function."""
    captured = {}

    def fake_signal(sig, handler):
        if sig == _signal.SIGTERM:
            captured["handler"] = handler
        return None

    def fake_run(*a, **kw):
        raise SystemExit  # break out of main() cleanly

    import piper_multi_server as pms
    with patch.object(_signal, "signal", side_effect=fake_signal), \
         patch("flask.Flask.run", side_effect=fake_run):
        with pytest.raises(SystemExit):
            import sys
            old_argv = sys.argv[:]
            sys.argv = ["piper_multi_server.py", "--admin-token", "shutdown-tok"]
            try:
                pms.main()
            finally:
                sys.argv = old_argv

    assert "handler" in captured, "SIGTERM handler was not registered"
    return captured["handler"]


class TestGracefulShutdown:

    def test_sigterm_handler_is_registered(self, main_handler):
        """If main() ever stops registering the handler, this fails."""
        assert callable(main_handler)

    def test_handler_drains_active_calls(self, main_handler):
        """When invoked, the handler must call delete_call for every
        active call so peers receive a BYE."""
        from speech_pipeline.telephony import call_state

        # Seed two fake calls.
        c1 = call_state.create_call("sub", "acc", "pbx")
        c2 = call_state.create_call("sub", "acc", "pbx")
        ids = [c1.call_id, c2.call_id]
        try:
            deleted = []

            def fake_delete(cid):
                deleted.append(cid)
                # Simulate teardown without actually killing mixers
                # (avoids cascading thread cleanup).
                call_state._calls.pop(cid, None)

            with patch.object(call_state, "delete_call",
                              side_effect=fake_delete):
                # Replace os.kill so the handler doesn't actually
                # signal the test process.
                import os
                with patch.object(os, "kill"):
                    # SIG_DFL re-arm is fine — we never re-fire.
                    try:
                        main_handler(_signal.SIGTERM, None)
                    except SystemExit:
                        pass

            for cid in ids:
                assert cid in deleted, (
                    f"Graceful shutdown skipped call {cid}"
                )
        finally:
            for cid in ids:
                call_state._calls.pop(cid, None)
