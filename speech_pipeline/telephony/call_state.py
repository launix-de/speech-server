"""In-memory call state.

A call = a conference = a ``ConferenceMixer``.

- ``add_source(stage)`` → pipes audio into the mix
- ``add_sink(stage, mute_source=id)`` → receives mix-minus audio

All format conversion is automatic via ``pipe()``.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from speech_pipeline.ConferenceMixer import ConferenceMixer
from .id_scope import scoped_id

_LOGGER = logging.getLogger("telephony.call")

_calls: Dict[str, "Call"] = {}

MIXER_SAMPLE_RATE = 48000


class Call:
    """A live conference backed by ConferenceMixer."""

    def __init__(self, subscriber_id: str, account_id: str, pbx_id: str,
                 caller: str = "", callee: str = "",
                 direction: str = "inbound",
                 events: Optional[Dict[str, str]] = None):
        self.call_id = scoped_id(account_id, "call", entropy_bytes=12)
        self.subscriber_id = subscriber_id
        self.account_id = account_id
        self.pbx_id = pbx_id
        self.caller = caller
        self.callee = callee
        self.direction = direction
        self.status = "active"
        self.created_at = time.time()
        self.events = events or {}
        self.stt_pipeline_id: Optional[str] = None

        self.mixer = ConferenceMixer(name=self.call_id,
                                     sample_rate=MIXER_SAMPLE_RATE,
                                     frame_samples=960)  # 20ms@48kHz — aligns with RTP timing
        # Auto-remove call from registry when mixer idles out
        self.mixer.on_idle_cancel = lambda _m: _auto_cleanup(self.call_id)
        self._participants: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._start()

    def _start(self) -> None:
        def _run():
            _LOGGER.info("Conference started: %s", self.call_id)
            try:
                self.mixer.run()
            except Exception as e:
                _LOGGER.warning("Conference %s error: %s", self.call_id, e)
            finally:
                self.status = "completed"
                _LOGGER.info("Conference stopped: %s", self.call_id)

        self._thread = threading.Thread(
            target=_run, daemon=True, name=f"conf-{self.call_id}")
        self._thread.start()

    def register_participant(self, pid: str, **meta) -> None:
        with self._lock:
            self._participants[pid] = meta

    def unregister_participant(self, pid: str) -> Optional[dict]:
        with self._lock:
            return self._participants.pop(pid, None)

    def get_participant(self, pid: str) -> Optional[dict]:
        """Get participant metadata (direct reference, not a copy)."""
        with self._lock:
            return self._participants.get(pid)

    def list_participants(self, type_filter: str = None) -> List[dict]:
        with self._lock:
            result = []
            for k, v in self._participants.items():
                if type_filter and v.get("type") != type_filter:
                    continue
                result.append({"id": k, **v})
            return result

    def end(self) -> None:
        self.mixer.cancel()
        with self._lock:
            self._participants.clear()
        self.status = "completed"
        _LOGGER.info("Call %s ended", self.call_id)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "subscriber_id": self.subscriber_id,
            "account_id": self.account_id,
            "pbx_id": self.pbx_id,
            "caller": self.caller,
            "callee": self.callee,
            "direction": self.direction,
            "status": self.status,
            "participants": self.list_participants(),
            "created_at": self.created_at,
        }


def create_call(subscriber_id: str, account_id: str, pbx_id: str,
                **kwargs: Any) -> Call:
    call = Call(subscriber_id, account_id, pbx_id, **kwargs)
    _calls[call.call_id] = call
    return call


def _auto_cleanup(call_id: str) -> None:
    """Called by ConferenceMixer when it auto-cancels (idle timeout).

    Removes the call from the registry + any associated legs.
    """
    _LOGGER.info("Auto-cleanup call %s (idle timeout)", call_id)
    call = _calls.pop(call_id, None)
    if not call:
        return
    call.status = "completed"
    # Hang up any leftover legs
    from . import leg as leg_mod
    for lg in list(leg_mod.list_legs()):
        if lg.call_id == call_id:
            try:
                lg.status = "completed"
                leg_mod.delete_leg(lg.leg_id)
            except Exception:
                pass

def get_call(call_id: str) -> Optional[Call]:
    return _calls.get(call_id)

def delete_call(call_id: str) -> bool:
    call = _calls.get(call_id)
    if not call:
        return False

    pipe_executor = getattr(call, "pipe_executor", None)
    if pipe_executor:
        try:
            pipe_executor.shutdown()
        except Exception:
            pass

    from . import webclient as webclient_mod
    webclient_mod.close_call_sessions(call_id)

    # Hang up all SIP legs associated with this call
    from . import leg as leg_mod
    for lg in leg_mod.list_legs():
        if lg.call_id == call_id:
            try:
                lg.status = "completed"
                leg_mod.delete_leg(lg.leg_id)
            except Exception:
                pass
    call.end()
    _calls.pop(call_id, None)
    return True

def list_calls(account_id: Optional[str] = None) -> List[Call]:
    if account_id:
        return [c for c in _calls.values() if c.account_id == account_id]
    return list(_calls.values())
