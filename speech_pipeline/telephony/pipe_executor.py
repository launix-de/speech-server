"""DSL pipe executor for telephony calls.

Parses DSL pipes like::

    sip:leg-abc{"completed":"/hook"} -> call:call-xyz -> sip:leg-abc
    play:https://example.com/hold.mp3{"loop":true,"volume":50} -> call:call-xyz
    tts:de_DE-thorsten-medium{"text":"Willkommen"} -> call:call-xyz

Elements: ``type:id{json_params}``
Separator: ``->`` (or ``|`` for backward compat)

All SIP bridging uses add_source + add_sink (proven pattern from e2e97ca).
Play uses the same loop logic as _cmd_play (proven to work).
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger("telephony.pipe-executor")


# ---------------------------------------------------------------------------
# DSL Parser
# ---------------------------------------------------------------------------

def parse_dsl(dsl: str) -> List[Tuple[str, str, dict]]:
    """Parse a DSL pipe string into elements.

    Returns list of (type, id, params_dict) tuples.
    """
    elements = []
    pos = 0
    s = dsl.strip()

    while pos < len(s):
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
        if pos >= len(s):
            break
        if s[pos:pos+2] == '->':
            pos += 2
            continue
        if s[pos] == '|':
            pos += 1
            continue

        # Read type
        start = pos
        while pos < len(s) and s[pos] not in ':{| \t':
            if s[pos] == '-' and pos + 1 < len(s) and s[pos+1] == '>':
                break
            pos += 1
        typ = s[start:pos].strip()

        # Read id (after :)
        elem_id = ""
        if pos < len(s) and s[pos] == ':':
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in '{| \t':
                if s[pos] == '-' and pos + 1 < len(s) and s[pos+1] == '>':
                    break
                pos += 1
            elem_id = s[start:pos].strip()

        # Read optional JSON params
        params = {}
        if pos < len(s) and s[pos] == '{':
            json_start = pos
            depth = 0
            in_string = False
            escape = False
            while pos < len(s):
                ch = s[pos]
                if escape:
                    escape = False
                elif ch == '\\' and in_string:
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            pos += 1
                            break
                pos += 1
            json_str = s[json_start:pos]
            try:
                params = json.loads(json_str)
            except json.JSONDecodeError as e:
                _LOGGER.warning("Failed to parse JSON in DSL: %s", e)

        elements.append((typ, elem_id, params))

    return elements


# ---------------------------------------------------------------------------
# Pipe Executor
# ---------------------------------------------------------------------------

class CallPipeExecutor:
    """Manages DSL pipe execution for a single Call.

    Uses the SAME wiring as bridge_to_call (add_source + add_sink)
    and _cmd_play (AudioReader + loop + add_source) — proven patterns.
    """

    def __init__(self, call, tts_registry=None, subscriber=None):
        self.call = call
        self._tts_registry = tts_registry
        self._subscriber = subscriber
        self._stages: Dict[str, Any] = {}  # stage_id -> handle object
        self._lock = threading.Lock()

    def add_pipes(self, pipes: List[str]) -> List[dict]:
        """Parse and execute DSL pipe strings."""
        results = []
        for pipe_str in pipes:
            try:
                elements = parse_dsl(pipe_str)
                self._execute_pipe(elements)
                results.append({"pipe": pipe_str, "ok": True})
            except Exception as e:
                _LOGGER.warning("Pipe failed: %s — %s", pipe_str, e, exc_info=True)
                results.append({"pipe": pipe_str, "ok": False, "error": str(e)})
        return results

    def kill_stage(self, stage_id: str) -> bool:
        """Cancel a named stage and clean up."""
        with self._lock:
            handle = self._stages.pop(stage_id, None)
        if not handle:
            return False

        # Signal stop event (for play loops)
        stop = getattr(handle, '_stop', None)
        if stop:
            stop.set()

        # Kill current mixer source
        src_id = getattr(handle, '_current_src', None)
        if src_id:
            try:
                self.call.mixer.kill_source(src_id)
            except Exception:
                pass

        _LOGGER.info("Killed stage %s", stage_id)
        return True

    def kill_all_play(self) -> int:
        """Kill all play stages. Returns count of killed stages."""
        killed = 0
        with self._lock:
            play_keys = [k for k in self._stages if k.startswith("play:")]
        for key in play_keys:
            if self.kill_stage(key):
                killed += 1
        return killed

    def list_stages(self) -> List[dict]:
        with self._lock:
            return [{"id": k} for k in self._stages]

    # ------------------------------------------------------------------
    # Pipe execution
    # ------------------------------------------------------------------

    def _execute_pipe(self, elements: List[Tuple[str, str, dict]]):
        """Execute a parsed DSL pipe.

        Recognizes these patterns:
        - sip:LEG -> call:CALL -> sip:LEG  (bidirectional SIP bridge)
        - play:URL{params} -> call:CALL     (hold music / playback)
        - tts:VOICE{text} -> call:CALL      (text-to-speech)
        - call:CALL -> stt:LANG{cb}         (speech-to-text)
        """
        if not elements:
            return

        types = [e[0] for e in elements]

        # Pattern: sip -> call -> sip (bidirectional bridge)
        if types == ["sip", "call", "sip"]:
            self._pipe_sip_bridge(elements)

        # Pattern: play -> call (hold music)
        elif types == ["play", "call"]:
            self._pipe_play(elements)

        # Pattern: tts -> call (TTS announcement)
        elif types == ["tts", "call"]:
            self._pipe_tts(elements)

        # Pattern: call -> stt (speech recognition)
        elif len(types) >= 2 and types[0] == "call" and "stt" in types:
            self._pipe_stt(elements)

        else:
            _LOGGER.warning("Unrecognized pipe pattern: %s", " -> ".join(types))

    # ------------------------------------------------------------------
    # Pattern: sip -> call -> sip (bridge leg into conference)
    # Uses EXACTLY the same code as bridge_to_call (proven at e2e97ca)
    # ------------------------------------------------------------------

    def _pipe_sip_bridge(self, elements):
        """Bridge a SIP leg into the conference."""
        from . import leg as leg_mod

        sip_typ, leg_id, sip_params = elements[0]
        call_typ, call_id, call_params = elements[1]

        leg = leg_mod.get_leg(leg_id)
        if not leg:
            raise ValueError(f"Leg {leg_id} not found")
        if not leg.voip_call:
            raise ValueError(f"Leg {leg_id} has no SIP call")

        # Attach webhook callbacks
        if sip_params:
            leg.callbacks.update(sip_params)

        # Mark as bridged (sip_listener._wait_for_bridge checks leg.call_id)
        leg.call_id = self.call.call_id

        # Use the proven bridge_to_call function directly
        leg_mod.bridge_to_call(leg, self.call)

        _LOGGER.info("SIP bridge: %s -> call:%s (via bridge_to_call)",
                     leg_id, self.call.call_id)

    # ------------------------------------------------------------------
    # Pattern: play -> call (hold music / audio playback)
    # Uses EXACTLY the same logic as _cmd_play (proven loop + volume)
    # ------------------------------------------------------------------

    def _pipe_play(self, elements):
        """Play audio into the conference."""
        from speech_pipeline.AudioReader import AudioReader

        play_typ, url, params = elements[0]
        if not url:
            raise ValueError("play requires URL")

        loop = params.get("loop", False)
        volume = params.get("volume", 100)
        stage_id = f"play:{url}"

        # Stop event for kill_stage
        stop_event = threading.Event()
        handle = type('PlayHandle', (), {
            '_stop': stop_event,
            '_current_src': None,
        })()

        with self._lock:
            self._stages[stage_id] = handle

        def _run():
            mixer = self.call.mixer
            try:
                iterations = 0
                max_iter = (int(loop) if isinstance(loop, (int, float)) and loop > 1
                            else (999999 if loop else 1))

                while (iterations < max_iter
                       and not stop_event.is_set()
                       and not mixer.cancelled):
                    reader = AudioReader(url, chunk_seconds=0.5)
                    source_stage = reader

                    if volume != 100:
                        from speech_pipeline.GainStage import GainStage
                        gain = GainStage(reader.output_format.sample_rate,
                                         float(volume) / 100.0)
                        reader.pipe(gain)
                        source_stage = gain

                    src_id = mixer.add_source(source_stage)
                    handle._current_src = src_id

                    mixer.wait_source(src_id)

                    if stop_event.is_set():
                        mixer.kill_source(src_id)
                    else:
                        mixer.remove_source(src_id)
                    iterations += 1
            except Exception as e:
                _LOGGER.warning("Play failed: %s", e)
            finally:
                with self._lock:
                    self._stages.pop(stage_id, None)

        threading.Thread(target=_run, daemon=True,
                         name=f"play-{secrets.token_urlsafe(4)}").start()

        _LOGGER.info("Play started: %s (loop=%s, volume=%s)", url, loop, volume)

    # ------------------------------------------------------------------
    # Pattern: tts -> call (TTS announcement)
    # ------------------------------------------------------------------

    def _pipe_tts(self, elements):
        """Speak text into the conference."""
        tts_typ, voice_name, params = elements[0]
        voice_name = voice_name or "de_DE-thorsten-medium"
        text = params.get("text", "")
        if not text:
            raise ValueError("tts requires text param")
        if not self._tts_registry:
            raise ValueError("TTS registry not available")

        from speech_pipeline.TTSProducer import TTSProducer
        voice = self._tts_registry.ensure_loaded(voice_name)
        syn = self._tts_registry.create_synthesis_config(voice, {})
        producer = TTSProducer(voice, syn, text, sentence_silence=0.0)

        def _run():
            mixer = self.call.mixer
            try:
                src_id = mixer.add_source(producer)
                mixer.wait_source(src_id)
                mixer.remove_source(src_id)
            except Exception as e:
                _LOGGER.warning("TTS failed: %s", e)

        threading.Thread(target=_run, daemon=True,
                         name=f"tts-{secrets.token_urlsafe(4)}").start()

        _LOGGER.info("TTS started: %s (voice=%s)", text[:40], voice_name)

    # ------------------------------------------------------------------
    # Pattern: call -> stt (speech recognition)
    # ------------------------------------------------------------------

    def _pipe_stt(self, elements):
        """Start STT on the conference output."""
        call_typ, call_id, call_params = elements[0]

        # Find stt element
        stt_elem = None
        webhook_elem = None
        for typ, eid, params in elements[1:]:
            if typ == "stt":
                stt_elem = (typ, eid, params)
            elif typ == "webhook":
                webhook_elem = (typ, eid, params)

        if not stt_elem:
            raise ValueError("No stt element found")

        lang = stt_elem[1] or "de"
        stt_params = stt_elem[2]

        from speech_pipeline.WhisperSTT import WhisperTranscriber
        stt = WhisperTranscriber(
            model_name=stt_params.get("model", "base"),
            language=lang)

        # Build webhook sink if present
        if webhook_elem:
            url = webhook_elem[1]
            from speech_pipeline.WebhookSink import WebhookSink
            bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
            sink = WebhookSink(url, bearer_token=bearer)
            stt.pipe(sink)
            self.call.mixer.add_sink(sink, mute_source=None)
        elif stt_params.get("on_receive"):
            url = stt_params["on_receive"]
            from speech_pipeline.WebhookSink import WebhookSink
            bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
            sink = WebhookSink(url, bearer_token=bearer)
            stt.pipe(sink)
            self.call.mixer.add_sink(sink, mute_source=None)
        else:
            self.call.mixer.add_sink(stt, mute_source=None)

        _LOGGER.info("STT started: lang=%s", lang)
