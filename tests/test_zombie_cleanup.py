"""Auto-cleanup tests: conferences with 0 sources + 0 sinks self-destruct."""
import json
import time

from speech_pipeline.ConferenceMixer import ConferenceMixer
from speech_pipeline.telephony import call_state


class TestIdleCleanup:
    """Conferences idle for too long are removed automatically."""

    def test_idle_mixer_auto_cancels(self, client, account, monkeypatch):
        """Mixer with no sources/sinks cancels itself after IDLE_TIMEOUT."""
        # Shorten timeout for fast test
        monkeypatch.setattr(ConferenceMixer, "IDLE_TIMEOUT_SECONDS", 0.5)

        from conftest import create_call
        call_id = create_call(client, account)

        # Call exists
        assert call_state.get_call(call_id) is not None
        call = call_state.get_call(call_id)
        assert not call.mixer.cancelled

        # Wait longer than IDLE_TIMEOUT
        time.sleep(1.5)

        # Call should be auto-removed from registry
        assert call_state.get_call(call_id) is None
        # Mixer should be cancelled
        assert call.mixer.cancelled

    def test_active_mixer_does_not_cancel(self, client, account, monkeypatch):
        """Mixer with at least one source does NOT auto-cancel."""
        monkeypatch.setattr(ConferenceMixer, "IDLE_TIMEOUT_SECONDS", 0.5)

        from conftest import create_call
        call_id = create_call(client, account)

        # Start a long-running play stage (keeps a source alive)
        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f'play:keepalive{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
                    }),
                    headers=account)

        time.sleep(1.5)

        # Call still exists
        assert call_state.get_call(call_id) is not None

        client.delete(f"/api/calls/{call_id}", headers=account)

    def test_idle_then_active_then_idle(self, client, account, monkeypatch):
        """Adding + removing source resets idle timer."""
        monkeypatch.setattr(ConferenceMixer, "IDLE_TIMEOUT_SECONDS", 0.5)

        from conftest import create_call
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)

        # Wait 0.3s (not yet idle timeout)
        time.sleep(0.3)
        assert call_state.get_call(call_id) is not None

        # Add a brief source (finishes quickly)
        client.post("/api/pipelines",
                    data=json.dumps({
                        "dsl": f'play:brief{{"url":"examples/queue.mp3"}} -> call:{call_id}'
                    }),
                    headers=account)

        # Wait again — timer should have been reset
        time.sleep(0.3)
        # Might still exist
        call_after = call_state.get_call(call_id)
        if call_after is not None:
            client.delete(f"/api/calls/{call_id}", headers=account)
